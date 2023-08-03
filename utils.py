import os
import time
import logging
from imp import reload

import numpy as np
import pandas as pd

def build_log(cfg):
    reload(logging)
    logging_path = os.path.join(cfg.output_dir, f"train_{cfg.ver}_{time.strftime('%m%d_%H%M', time.localtime())}_seed_{cfg.seed}.log")
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s %(message)s',
        datefmt = '%H:%M:%S',
        handlers = [
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

def print_log(cfg, message):
    if cfg.use_log:
        logging.info(message)
    else:
        print(message)

def metric_fn(y_pred, y_true, return_component_scores = False):
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
        y_true = y_true.values
    if not return_component_scores:
        return np.sqrt(((y_pred - y_true)**2).mean(axis = 0)).mean()
    else:
        component_scores = np.sqrt(((y_pred - y_true)**2).mean(axis = 0))
        return component_scores.mean(), component_scores[0], component_scores[1]
