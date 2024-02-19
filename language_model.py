import torch
from torch import nn

from src import bootstrap

from transformers import BertConfig


class maskedLanguageModel(nn.Module):
    def __init__(
            self,
            dim,
            vocab_size
    ):
        super().__init__()

        self.linear = nn.Linear(dim=dim, vocab_size=vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x))

class rbLanguageModel(nn.Module):
    def __init__(
            self,
            model,
            config
    ):
        super().__init__()
 
        self.rb_config = config # config

        self.dim = self.rb_config.bert_config['hidden_size']
        self.vocab_size = self.rb_config.bert_config['vocab_max_size']

        self.recipebuild = model(config) # recipebuild
        
        self.mask_lm = maskedLanguageModel(self.dim, self.vocab_size)

    def forward(self, x):

        x = self.recipebuild(x) # last_hidden_state

        return self.mask_lm(x) # linear transformation & softmax
