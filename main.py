import os, argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from custom_config import Config, set_random_seed
from utils import print_log
from trainer import Trainer

if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description = 'Fine-tuning the LongFinBERT model on 10-X filings...')
    
    # General settings
    parser.add_argument('--seed', type = int, default = 1, help = 'The random state.')
    parser.add_argument('--ver', type = str, default = 'v1a', help = 'The name of the current version.')
    parser.add_argument('--use_log', type = int, default = 0, help = 'Whether we log the training process or not, only takes 0 or 1.')
    parser.add_argument('--use_tqdm', type = int, default = 0, help = 'Whether we use loop tracking or not, only takes 0 or 1.')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'The training device.')

    # Model
    parser.add_argument('--backbone', type = str, default = '.', help = 'The checkpoint of the tokenizer.')
    
    # Data
    parser.add_argument('--max_len', type = int, default = 10_000, help = 'The maximum sequence length.')
    
    # Training
    parser.add_argument('--apex', type = int, default = 1, help = 'Whether or not we train with mixed precision.')
    parser.add_argument('--nepochs', type = int, default = 5, help = 'The number of training epochs.')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'The batch size.')
    
    # Optimizer
    parser.add_argument('--lr', type = float, default = 2e-5, help = 'The training learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'The coefficient for L2-regularization.')
    parser.add_argument('--min_lr', type = float, default = 1e-6, help = 'The minimum training learning rate.')
    
    # Schduler
    parser.add_argument('--scheduler_type', type = str, default = 'cosine', help = 'The type of the scheduler.')
    parser.add_argument('--num_warmup_steps', type = float, default = 0., help = 'The proportion of warm-up steps.')
    
    # Paths
    parser.add_argument('--train_data_dir', type = str, default = '.', help = 'The training data directory.')
    parser.add_argument('--valid_data_dir', type = str, default = '.', help = 'The validating data directory.')
    parser.add_argument('--test_data_dir', type = str, default = '.', help = 'The testing data directory.')
    
    args = parser.parse_args()
    
    cfg = Config(args)
    
    if cfg.use_log:
        build_log(cfg)
        
    Trainer(cfg).fit()
