import os, copy, math
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

from huggingface_hub import hf_hub_download

from custom_config import LongBERTConfig
    
class LongBERTOutput(object):
    def __repr__(self):
        return f"LongBERTOutput: {self.__dict__}"

def clone(module, N):
    '''
    This function clones a module by N times
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def vanilla_attention(query, key, value, key_padding_mask = None, attn_mask = None, dropout = None):
    assert len(query.shape) == 4, "The 'query' tensor should have 4 dimensions (batch_size, n_head, seq_len, dim)"
    assert len(key.shape) == 4, "The 'key' tensor should have 4 dimensions (batch_size, n_head, seq_len, dim)"
    assert len(value.shape) == 4, "The 'value' tensor should have 4 dimensions (batch_size, n_head, seq_len, dim)"
    
    batch_size, n_head, target_seq_len, dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    source_seq_len = key.shape[2]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim)
    
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, source_seq_len)
        if isinstance(key_padding_mask[0, 0, 0, 0].item(), bool):
            scores = scores.masked_fill_(key_padding_mask, float('-inf'))
        elif isinstance(key_padding_mask[0, 0, 0, 0].item(), float):
            scores -= key_padding_mask
        else:
            raise TypeError("The 'key_padding_mask' can only be either a boolean tensor or a float tensor")
    
    if attn_mask is not None:
        assert len(attn_mask.shape) in [2, 3], "The 'attn_mask' tensor must be a 2-D or 3-D tensor."
        assert attn_mask.shape[-1] == source_seq_len
        assert attn_mask.shape[-2] == target_seq_len
        
        if len(attn_mask.shape) == 2:
            # The mask is applied to all entries in a batch, with all heads
            attn_mask = attn_mask.view(1, 1, target_seq_len, source_seq_len)
        else:
            attn_mask = attn_mask.view(batch_size, -1, target_seq_len, source_seq_len)
            
        if isinstance(attn_mask[0, 0, 0, 0].item(), bool):
            scores = scores.masked_fill_(attn_mask, float('-inf'))
        elif isinstance(attn_mask[0, 0, 0, 0].item(), float):
            scores -= attn_mask
        else:
            raise TypeError("The 'attn_mask' can only be either a boolean tensor or a float tensor")
    
    attn_output_weights = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        attn_output_weights = F.dropout(attn_output_weights, p = dropout, training = True)
        
    return torch.matmul(attn_output_weights, value), attn_output_weights

def dilated_attention(query, key, value, key_padding_mask = None, attn_mask = None, dropout = None):
    assert len(query.shape) == 5, "The 'query' tensor should have 4 dimensions (batch_size, n_head, n_segment, segment_size, dim)"
    assert len(key.shape) == 5, "The 'key' tensor should have 4 dimensions (batch_size, n_head, n_segment, segment_size, dim)"
    assert len(value.shape) == 5, "The 'value' tensor should have 4 dimensions (batch_size, n_head, n_segment, segment_size, dim)"
    
    batch_size, n_head, n_segment, target_seq_len, dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3], query.shape[4]
    source_seq_len = key.shape[3]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim)
    
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(batch_size, 1, n_segment, 1, source_seq_len)
        if isinstance(key_padding_mask[0, 0, 0, 0, 0].item(), bool):
            scores = scores.masked_fill_(key_padding_mask, float('-inf'))
        elif isinstance(key_padding_mask[0, 0, 0, 0, 0].item(), float):
            scores -= key_padding_mask
        else:
            raise TypeError("The 'key_padding_mask' can only be either a boolean tensor or a float tensor")
            
    if attn_mask is not None:
        assert len(attn_mask.shape) in [3, 4], "The 'attn_mask' tensor must be a 3-D or 4-D tensor."
        assert attn_mask.shape[-1] == source_seq_len
        assert attn_mask.shape[-2] == target_seq_len
        
        if len(attn_mask.shape) == 3:
            # The mask is applied to all entries in a batch, with all heads
            attn_mask = attn_mask.view(1, 1, n_segment, target_seq_len, source_seq_len)
        else:
            attn_mask = attn_mask.view(batch_size, -1, n_segment, target_seq_len, source_seq_len)
            
        if isinstance(attn_mask[0, 0, 0, 0, 0].item(), bool):
            scores = scores.masked_fill_(attn_mask, float('-inf'))
        elif isinstance(attn_mask[0, 0, 0, 0, 0].item(), float):
            scores -= attn_mask
        else:
            raise TypeError("The 'attn_mask' can only be either a boolean tensor or a float tensor")
    
    segment_weights = torch.exp(scores).sum(dim = -1)
    attn_output_weights = F.softmax(scores, dim = -1)
    attn_output_weights = torch.nan_to_num(attn_output_weights)
    
    if dropout is not None:
        attn_output_weights = F.dropout(attn_output_weights, p = dropout, training = True)
        
    return torch.matmul(attn_output_weights, value), attn_output_weights, segment_weights

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head, dropout = 0.1):
        super(MultiheadAttention, self).__init__()
        '''
        The implementation of the multihead attention mechanism
        Remarks:
            - We assume the query, key, and value have the same dimensionality (embedding_dim)
        '''
        assert embedding_dim % n_head == 0, "The embedding dimension should be divisible by the number of heads"
        self.d_proj = embedding_dim // n_head
        self.n_head = n_head
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        
    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None):
        batch_size, seq_len, embedding_dim = query.shape[0], query.shape[1], query.shape[2]
        
        # Projection
        query = self.q_proj(query).view(batch_size, -1, self.n_head, self.d_proj).transpose(1, 2)    # Shape: (batch_size, n_head, seq_len, d_proj)
        key = self.k_proj(key).view(batch_size, -1, self.n_head, self.d_proj).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, -1, self.n_head, self.d_proj).transpose(1, 2)
        
        # Attention
        attn_output, attn_output_weights = vanilla_attention(query, key, value, key_padding_mask = key_padding_mask, attn_mask = attn_mask, dropout = self.dropout)
        
        # Concatenation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        attn_output_weights = attn_output_weights.transpose(1, 2).contiguous().view(batch_size, seq_len, seq_len)
        
        return attn_output, attn_output_weights
    
class DilatedMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head, segment_size, dilated_rate, dropout = 0.1):
        super(DilatedMultiheadAttention, self).__init__()
        assert embedding_dim % n_head == 0, "The embedding dimension should be divisible by the number of heads"
        self.d_proj = embedding_dim // n_head
        self.n_head = n_head
        self.segment_size = segment_size
        self.dilated_rate = dilated_rate
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        
    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None):
        batch_size, seq_len, embedding_dim = query.shape[0], query.shape[1], query.shape[2]
        
        attn_output = []
        
        for j, (segment_size, dilated_rate) in enumerate(zip(self.segment_size, self.dilated_rate)):
            if seq_len % segment_size != 0:
                # If the seq_len is not divisible by segment_size, we pad the sequence
                add_padding_len = segment_size - seq_len % segment_size
                _seq_len = seq_len + add_padding_len
                padded = torch.zeros([batch_size, add_padding_len, embedding_dim], device = query.device, dtype = query.dtype)
                _query = torch.cat([query.contiguous(), padded], dim = 1)
                _key = torch.cat([key.contiguous(), padded], dim = 1)
                _value = torch.cat([value.contiguous(), padded], dim = 1)
                
                padded_mask = torch.zeros([batch_size, add_padding_len], device = key_padding_mask.device, dtype = key_padding_mask.dtype)
                _key_padding_mask = torch.cat([key_padding_mask.contiguous(), padded_mask], dim = 1)
            else:
                _seq_len = seq_len
                _query = query.contiguous()
                _key = key.contiguous()
                _value = value.contiguous()
                
                _key_padding_mask = key_padding_mask.contiguous()
                
            n_segment = _seq_len // segment_size
            
            # Break sequences into segments
            _query = _query.view(batch_size, n_segment, segment_size, -1)    # Shape: (batch_size, n_segment, segment_size, dim)
            _key = _key.view(batch_size, n_segment, segment_size, -1)
            _value = _value.view(batch_size, n_segment, segment_size, -1)
            
            # Apply dilation
            _query = _query[:,:,::dilated_rate,:]
            _key = _key[:,:,::dilated_rate,:]
            _value = _value[:,:,::dilated_rate,:]

            # Projection
            _query = self.q_proj(_query).view(batch_size, n_segment, -1, self.n_head, self.d_proj).permute(0, 3, 1, 2, 4)
            _key = self.q_proj(_key).view(batch_size, n_segment, -1, self.n_head, self.d_proj).permute(0, 3, 1, 2, 4)
            _value = self.q_proj(_value).view(batch_size, n_segment, -1, self.n_head, self.d_proj).permute(0, 3, 1, 2, 4)
            
            if key_padding_mask is not None:
                _key_padding_mask = _key_padding_mask.view(batch_size, n_segment, segment_size)[:,:,::dilated_rate]
            else:
                _key_padding_mask = None
            
            # Attention
            _attn_output, _attn_output_weights, _segment_weights = dilated_attention(_query, _key, _value, key_padding_mask = _key_padding_mask, attn_mask = attn_mask, dropout = self.dropout)
            # Shape of attn_output: (batch_size, n_head, n_segment, segment_size, dim)
            # Shape of attn_output_weights: (batch_size, n_head, n_segment, segment_size, segment_size)
            # Shape of segment_weights: (batch_size, n_head, n_segment, segment_size)
            # attn_output = attn_output * segment_weights.unsqueeze(-1)
            
            attn_output_resized = torch.zeros((batch_size, n_segment, segment_size, self.n_head, self.d_proj), 
                                              device = _attn_output.device, dtype = _attn_output.dtype)
            attn_output_resized[:,:,::dilated_rate,:,:] = _attn_output.contiguous().permute(0, 2, 3, 1, 4)
            _attn_output = attn_output_resized.contiguous()
            '''
            # Reconstruct the attention weights
            attn_output_weights_resized = torch.zeros((batch_size, self.n_head, n_segment, seq_len, seq_len), 
                                                      device = _attn_output_weights.device, dtype = _attn_output_weights.dtype)
            for i in range(0, seq_len, segment_size):
                _s = i
                _e = min(i + segment_size, seq_len)
                attn_output_weights_resized[:,:,:, slice(_s, _e, dilated_rate), slice(_s, _e, dilated_rate)] = _attn_output_weights.contiguous()
            _attn_output_weights = attn_output_weights_resized.contiguous()    # Shape: (batch_size, n_head, n_segment, segment_size, segment_size)
            
            # Reconstruct the segment weights
            segment_weights_resized = torch.zeros((batch_size, self.n_head, n_segment, seq_len),
                                                  device = _segment_weights.device, dtype = _segment_weights.dtype)
            
            for i in range(0, seq_len, segment_size):
                _s = i
                _e = min(i + segment_size, seq_len)
                segment_weights_resized[:,:,:, slice(_s, _e, dilated_rate)] = _segment_weights.contiguous()
            _segment_weights = segment_weights_resized.contiguous()    # Shape: (batch_size, n_head, n_segment, seq_len)
            '''
            # Concatenation
            _attn_output = _attn_output.contiguous().view(batch_size, n_segment, segment_size, embedding_dim)
            _attn_output = _attn_output.permute(0, 3, 1, 2).view(batch_size, embedding_dim, _seq_len).transpose(1, 2)
            
            if seq_len % segment_size != 0:
                _attn_output = _attn_output[:,:seq_len,:]
            
            if j == 0:
                attn_output = _attn_output / len(self.segment_size)
            else:
                attn_output += _attn_output / len(self.segment_size)
        
        return attn_output

class LongBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(LongBERTEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx = 0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = 1e-12, elementwise_affine = 1e-12)
        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)
        
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        finbert_model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')
        
        # Extract the pre-trained embeddings weight from FinBERT
        embeddings_state_dict = finbert_model.embeddings.state_dict()
        word_embeddings = embeddings_state_dict.pop('word_embeddings.weight')
        position_embeddings = embeddings_state_dict.pop('position_embeddings.weight')
        token_type_embeddings = embeddings_state_dict.pop('token_type_embeddings.weight')
        LayerNorm_weight = embeddings_state_dict.pop('LayerNorm.weight')
        LayerNorm_bias = embeddings_state_dict.pop('LayerNorm.bias')

        # Clone the positional_embeddings
        max_len = self.config.max_position_embeddings
        slices = list(range(0, max_len - 1, len(position_embeddings) - 1)) + [max_len - 2]
        slices = [(1, 1 + sl2 - sl1) for (sl1, sl2) in zip(slices[:-1], slices[1:])]
        position_embeddings = torch.cat([position_embeddings[:1]] + [position_embeddings[slice(*sl)] for sl in slices] + [position_embeddings[-1:]])

        # Load the embeddings weights into our model for initialization
        self.word_embeddings.load_state_dict({'weight': word_embeddings})
        self.position_embeddings.load_state_dict({'weight': position_embeddings})
        self.token_type_embeddings.load_state_dict({'weight': token_type_embeddings})
        self.LayerNorm.load_state_dict({'weight': LayerNorm_weight, 'bias': LayerNorm_bias})
        
    def forward(self, input_ids, token_type_ids, position_ids):
        word_embeddings = self.word_embeddings(input_ids)
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
        else:
            token_type_embeddings = 0
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)
    
class LongBERTLayer(nn.Module):
    def __init__(self, config):
        super(LongBERTLayer, self).__init__()        
        self.attention = DilatedMultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                                   config.segment_size, config.dilated_rate, dropout = config.attention_probs_dropout_prob)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps = 1e-12, elementwise_affine = True),
            nn.Dropout(p = config.hidden_dropout_prob, inplace = False),
        )
    
    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None):
        attn_output = self.attention(query, key, value, key_padding_mask = key_padding_mask, attn_mask = attn_mask)
        attn_output = self.output(attn_output)
        return attn_output
    
class LongBERTPooler(nn.Module):
    def __init__(self, config):
        super(LongBERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias = True)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_state):
        return self.activation(self.dense(hidden_state[:,0,:]))

class LongBERTEncoder(nn.Module):
    def __init__(self, config):
        super(LongBERTEncoder, self).__init__()
        self.layer = clone(LongBERTLayer(config), config.num_hidden_layers)    # For the base version, we use only 12 layers
        self.pooler = LongBERTPooler(config)
        self.longbert_output = LongBERTOutput()
        
    def forward(self, hidden_state, attention_mask = None, output_hidden_states = False):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else attention_mask
        hidden_states = tuple()
        for i, layer in enumerate(self.layer):
            hidden_state = layer(hidden_state, hidden_state, hidden_state, key_padding_mask = key_padding_mask)
            hidden_states = hidden_states + (hidden_state,)
        
        self.longbert_output.pooled_output = self.pooler(hidden_state)
        self.longbert_output.last_hidden_state = hidden_state
        
        if output_hidden_states:
            self.longbert_output.hidden_states = hidden_states
            
        return self.longbert_output
    
class LongBERTModel(nn.Module):
    def __init__(self, config = None):
        super(LongBERTModel, self).__init__()
        self.embeddings = LongBERTEmbeddings(config) if config is not None else None
        self.encoder = LongBERTEncoder(config) if config is not None else None
    
    @classmethod
    def from_config(self, config):
        self.config = config
        return self(config = config)
    
    @classmethod
    def from_pretrained(self, ckpt):
        model_ckpt = hf_hub_download(repo_id = ckpt, filename = 'pytorch_model.bin')
        # Load the config first
        model_config = LongBERTConfig.from_pretrained(ckpt)
        self = self(config = model_config)
        try:
            self.load_state_dict(torch.load(model_ckpt, map_location = torch.device('cpu')))
        except:
            self.load_state_dict(torch.load(model_ckpt, map_location = torch.device('cuda:0')))
        return self
        
    def save_pretrained(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
                
    def forward(self, input_ids, attention_mask = None, token_type_ids = None, output_hidden_states = False):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype = torch.long).view(1, -1).repeat_interleave(batch_size, dim = 0).to(input_ids.device)
        hidden_state = self.embeddings(input_ids, token_type_ids, position_ids)
        return self.encoder(hidden_state, attention_mask = attention_mask, output_hidden_states = output_hidden_states)
    
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = LongBERTModel(cfg.config)
        self.output = nn.Linear(cfg.config.hidden_size, cfg.config.vocab_size)
        
    def loss_fn(self, pred, true):
        mask = true != -100
        pred = pred[mask,:]
        true = true[mask]
        return nn.CrossEntropyLoss()(pred, true)
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        hidden_state = self.backbone(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        output = self.output(hidden_state)
        
        if labels is not None:
            loss = self.loss_fn(output, labels)
        else:
            loss = None
            
        return loss, output
