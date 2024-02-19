import random
import numpy as np
import copy
import os
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import bootstrap
from utils import *

class TensorData(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        self.len = len(self.labels)

    def __getitem__(self, index):
        return {"input_ids" : self.input_ids[index], "labels" : self.labels[index]}

    def __len__(self):
        return self.len

def collate_fn(batch): 
    collate_input_ids = []
    collate_labels = []

    max_len = rb_config.bert_config['max_len']
    # max_len = max([len(sample['input_ids']) for sample in batch])

    for sample in batch:        

        # padding
        diff = max_len - len(sample['input_ids'])
        
        pad = torch.ones(size= (diff,))

        collate_input_ids.append(torch.cat([sample['input_ids'].view([len(sample['input_ids'])]), pad], dim=0))
        collate_labels.append(torch.cat([sample['labels'].view([len(sample['labels'])]), pad], dim=0))

        input_ids = torch.stack(collate_input_ids)
        labels = torch.stack(collate_labels)
        
        # masking
        masking_strategy_config = rb_config.bert_config['masking_strategy']

        masking_prob = masking_strategy_config['masking_prob']
        full_mask = torch.randn(input_ids.shape) < masking_prob

        tokenizer = get_tokenizer(rb_config)
        special_tokens = list(tokenizer.added_tokens_decoder.keys())
        for tk in special_tokens:
            full_mask = full_mask & (input_ids != tk)

        random_prob = masking_strategy_config['random_prob']
        random_mask = torch.randn(input_ids.shape) < random_prob
        full_mask_with_random = full_mask & (random_mask)

        unchanged_prob = masking_strategy_config['unchanged_prob']
        unchanged_mask = torch.randn(input_ids.shape) < unchanged_prob
        full_mask_with_unchanged = full_mask & (unchanged_mask)

        full_mask_with_mask = full_mask & (~full_mask_with_random) & (~full_mask_with_unchanged)

        final_mask = input_ids.clone()
        num_random_tokens = full_mask_with_random.sum().item()
        random_tokens = torch.randint(0, rb_config.bert_config['vocab_max_size'], size=(num_random_tokens,))
        indices = torch.nonzero(full_mask_with_random, as_tuple=True)
        final_mask[indices] = random_tokens.float()

        mask_token = 4
        final_mask = final_mask.masked_fill_(full_mask_with_mask, mask_token)

        input_ids = final_mask
    
    return {'input_ids': input_ids, 'labels' : labels}

def MyDataLoader(data_path, config):

    seed_everything() # random seed

    rb_config = config # load config

    dataset = load_dataset(data_path) # load dataset

    tokenizer = get_tokenizer(rb_config) # load tokenizer

    tokens = tokenizer(dataset)['input_ids'] # tokenize the dataset
    
    # make list of labels tensor
    labels = []
    for v in tokens:
        labels.append(torch.tensor(v))

    # make list of input_ids tensor
    input_ids = copy.deepcopy(labels) 

    data_tensor = TensorData(input_ids, labels)
 
    data_loader = DataLoader(data_tensor, batch_size=rb_config.bert_config['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)

    return data_loader

if __name__ == "__main__":
    rb_config = bootstrap.recipebuildConfig(
        path = "/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
    ) 

    data_loader = MyDataLoader(data_path=rb_config.processed_data_folder + '/v3_ing_title_tag_sample/train.txt', config=rb_config)
