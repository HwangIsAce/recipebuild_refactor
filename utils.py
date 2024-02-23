import torch
from torch import nn

import numpy as np

import random
import os

from src.recipebuild_tokenizer import RBTokenizer
from src import bootstrap

from transformers import BertConfig

def get_rb_config():
    return bootstrap.recipebuildConfig(path="/home/jaesung/jaesung/research/recipebuild_retactor/config.json")

def seed_everything(seed = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_tokenizer(config_):
    tokenizer = RBTokenizer(config_)
    tokenizer.load()
    return tokenizer.tokenizer  

def load_dataset(path):
    f = open(path, 'r')
    lines = f.readlines()

    dataset = []
    for line in lines:
        dataset.append('[CLS]' + line.split('\n')[0])

    return dataset
