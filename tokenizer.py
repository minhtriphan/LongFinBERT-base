from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

class LongBERTTokenizer(AutoTokenizer):
    def __init__(self):
        super(AutoTokenizer, self).__init__()
    
    @classmethod
    def from_pretrained(self, ckpt, **kwargs):
        _special_tokens_map = hf_hub_download(repo_id = ckpt, filename = 'special_tokens_map.json')
        _tokenizer = hf_hub_download(repo_id = ckpt, filename = 'tokenizer.json')
        _tokenizer_config = hf_hub_download(repo_id = ckpt, filename = 'tokenizer_config.json')
        _vocab = hf_hub_download(repo_id = ckpt, filename = 'vocab.txt')

        path = '/'.join(_special_tokens_map.split('/')[:-1])
        return AutoTokenizer.from_pretrained(path, **kwargs)
