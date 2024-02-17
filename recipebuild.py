from src import bootstrap  
from src.recipebuild_tokenizer import RBTokenizer
from dataloader import *

import torch
from torch import nn

import math
import random

from transformers import BertConfig

# config
config = BertConfig()

rb_config = bootstrap.recipebuildConfig(
    path= "/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
)

# embedding

class TokenEmbedding(nn.Embedding) :
    def __init__(self, vocab_size, embed_size = 512) :
        super().__init__(vocab_size, embed_size, padding_idx = 0)

class PositionalEmbedding(nn.Module) :
    def __init__(self, d_model, max_len = rb_config.bert_config['max_len']) :
        super().__init__()
        
        # compute positional encoding in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.required_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x) :
        return self.pe[:, :x.size(1)]

class rbEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            dropout=0.1,
    ):
        super().__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        
    def forward(self, x):
        emb_x = self.token(x) + self.position(x)
        return self.dropout(emb_x)

class recipeBuild(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            dropout=0.1,
    ):
        super().__init__()
        self.embedding = rbEmbedding(vocab_size, embed_size, dropout=0.1)

    def forward(self, x):
        x = x['input_ids'].long()

        x = self.embedding(x)

        return x
        

if __name__ == "__main__":
   
    sample_train_data_path = rb_config.processed_data_folder + '/v3_ing_title_tag_sample/train.txt'

    sample_train_loader = MyDataLoader(data_path=sample_train_data_path)

    model = recipeBuild(vocab_size=rb_config.bert_config['vocab_max_size'],
                        embed_size=config.hidden_size,
                        dropout=config.hidden_dropout_prob)

    for batch in sample_train_loader:
        logits = model(batch)

        import IPython; IPython.embed(colosr="Linux"); exit(1)


