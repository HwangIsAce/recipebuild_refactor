import torch
from torch import nn

from src import bootstrap

from transformers import BertConfig



class maskedLanguageModel(nn.Module):
    def __init__(
            self,
            dim
    ):
        super().__init__()

        self.config = BertConfig()

        self.rb_config = bootstrap.recipebuildConfig(
            path="/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
        )
        self.linear = nn.Linear(dim, self.rb_config.bert_config['vocab_max_size'])
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class rbLanguageModel(nn.Module):
    def __init__(
            self,
            model
    ):
        super().__init__()
        self.config = BertConfig()

        self.rb_config = bootstrap.recipebuildConfig(
            path="/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
        )

        self.recipebuild = model(vocab_size=self.rb_config.bert_config['vocab_max_size'],
                        embed_size=self.config.hidden_size,
                        dim=self.config.hidden_size,
                        depth=self.rb_config.bert_config['num_hidden_layer'],
                        heads=self.rb_config.bert_config['num_heads'],
                        dim_head=self.config.hidden_size / self.rb_config.bert_config['num_heads'],
                        attn_dropout=self.config.hidden_dropout_prob,
                        ff_dropout=self.config.hidden_dropout_prob,
                        emb_dropout=self.config.hidden_dropout_prob)
        
        self.mask_lm = maskedLanguageModel(self.config.hidden_size)

    def forward(self, x):
        x = self.recipebuild(x)
        return self.mask_lm(x)
