import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class LongFinBERTDataset(Dataset):
    def __init__(self, cfg, mode = 'train'):
        self.cfg = cfg
        if mode == 'train':
            self.dataset = load_from_disk(cfg.train_data_dir)
        elif mode == 'valid':
            self.dataset = load_from_disk(cfg.valid_data_dir)
    
    def _tokenize(self, text):
        return self.cfg.tokenizer(text, 
                                  padding = False, 
                                  max_length = self.cfg.max_len - 2, 
                                  truncation = True, 
                                  add_special_tokens = False)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        tokenized_text = self._tokenize(text)
        
        return {
            'input_ids': np.array(tokenized_text['input_ids']),
        }
    
class Collator(object):
    def __init__(self, cfg, mlm_probability = 0.15):
        self.cfg = cfg
        self.mlm_probability = mlm_probability
        
    def random_masking(self, input_ids):
        seq_len = len(input_ids)
        np.random.seed(self.cfg.seed)
        chosen_indexes = np.random.choice(np.arange(seq_len), int(self.mlm_probability * seq_len), replace = False)
        # Generate labels
        labels = -np.ones_like(input_ids) * 100
        labels[chosen_indexes] = input_ids[chosen_indexes]
        # Mask input_ids
        input_ids[chosen_indexes] = self.cfg.tokenizer.mask_token_id
        return input_ids, labels
    
    def padding(self, input_ids, labels):
        # Add special tokens
        input_ids = [self.cfg.tokenizer.cls_token_id] + input_ids.tolist() + [self.cfg.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels.tolist() + [-100]
        
        if len(input_ids) < self.cfg.max_len:
            attention_mask = attention_mask + [0] * (self.cfg.max_len - len(input_ids))
            labels = labels + [-100] * (self.cfg.max_len - len(input_ids))
            input_ids = input_ids + [self.cfg.tokenizer.pad_token_id] * (self.cfg.max_len - len(input_ids))
        
        return input_ids, attention_mask, labels
    
    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []
        
        for i, item in enumerate(batch):
            _input_ids, _labels = self.random_masking(item['input_ids'])
            batch_input_ids, batch_attention_mask, batch_labels = self.padding(_input_ids, _labels)
            
            input_ids.append(batch_input_ids)
            attention_mask.append(batch_attention_mask)
            labels.append(batch_labels)
            
        input_ids = torch.tensor(input_ids, dtype = torch.long)
        attention_mask = torch.tensor(attention_mask, dtype = torch.long)
        labels = torch.tensor(labels, dtype = torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
