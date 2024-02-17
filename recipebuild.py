from src import bootstrap  
from src.recipebuild_tokenizer import RBTokenizer

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
    ):
        super().__init__()
        vocab_size = rb_config.bert_config['vocab_max_size'] 
        embed_size = config.hidden_size # 768
        dropout = config.hidden_dropout_prob # 0.1

        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = nn.Embedding(3, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, segment_label):
        emb_x = self.token(x) + self.position(x) + self.segment(segment_label)
        return self.dropout(emb_x)

class recipeBuild(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, x):
        pass
        

if __name__ == "__main__":
   

    import IPython; IPython.embed(colosr="Linux"); exit(1)


