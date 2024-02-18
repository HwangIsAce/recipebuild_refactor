from src import bootstrap  
from src.recipebuild_tokenizer import RBTokenizer
from dataloader import *

import torch
from torch import nn, einsum

import math
import random

from einops import rearrange

from transformers import BertConfig

# config
config = BertConfig()

rb_config = bootstrap.recipebuildConfig(
    path= "/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
)

############################################################################
def FeedForward(dim, mult=4, dropout=0.):

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim*mult*2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim*mult*2, dim)
    )
    

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=12,
            dropout=0.,
    ):
        super().__init__()
        inner_dim = int(dim_head) * heads
        self.heads = heads

        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x):
        h = self.heads
        
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q,k,v))
        q = q * self.scale
        
        sim = einsum('b h i d, b h j d-> b h i j', q,k)

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d-> b h i d', dropped_attn,v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out

# transformer
class Transformer(nn.Module):
    def __init__(
            self,
            depth,
            dim,
            heads,
            dim_head,
            attn_dropout,
            ff_dropout,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout)
            ]))

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out = attn(x)
            post_softmax_attns.append(attn_out)
            
            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
                return x
            
        return x, torch.stack(post_softmax_attns)

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
            emb_dropout=0.1,
    ):
        super().__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=emb_dropout)
        self.embed_size = embed_size
        
    def forward(self, x):

        emb_x = self.token(x) + self.position(x)

        return self.dropout(emb_x)

##############################################################################

class recipeBuild(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            dim,
            depth,
            heads,
            dim_head,
            attn_dropout,
            ff_dropout,
            emb_dropout,
    ):
        super().__init__()
        self.embedding = rbEmbedding(vocab_size, embed_size, emb_dropout=emb_dropout)

        self.encoder = Transformer(
            depth = depth,
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

    def forward(self, x):
        x = x['input_ids'].long()

        x = self.embedding(x)

        x = self.encoder(x)

        return x
        

if __name__ == "__main__":

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
   
    sample_train_data_path = rb_config.processed_data_folder + '/v3_ing_title_tag_sample/train.txt'

    sample_train_loader = MyDataLoader(data_path=sample_train_data_path)

    model = recipeBuild(vocab_size=rb_config.bert_config['vocab_max_size'],
                        embed_size=config.hidden_size,
                        dim=config.hidden_size,
                        depth=rb_config.bert_config['num_hidden_layer'],
                        heads=rb_config.bert_config['num_heads'],
                        dim_head=config.hidden_size / rb_config.bert_config['num_heads'],
                        attn_dropout=config.hidden_dropout_prob,
                        ff_dropout=config.hidden_dropout_prob,
                        emb_dropout=config.hidden_dropout_prob)

    for batch in sample_train_loader:
        logits = model(batch)

        import IPython; IPython.embed(colosr="Linux"); exit(1)


