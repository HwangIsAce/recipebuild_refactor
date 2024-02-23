import torch
from torch import nn
import tqdm

from einops import rearrange

from transformers import BertConfig

from recipebuild import recipeBuild
from language_model import rbLanguageModel
from dataloader import * 
from src import bootstrap

class rbTrainer(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
    
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        # config
        self.rb_config = bootstrap.recipebuildConfig(
            path = "/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
        )

        # model
        self.recipebuild = recipeBuild

        self.model = rbLanguageModel(model=self.recipebuild, config=self.rb_config).to(self.device)

        # data path
        self.data_path = self.rb_config.processed_data_folder 

        self.train_path = self.data_path + '/train.txt'
        self.val_path = self.data_path + '/val.txt'

        # data loader
        self.train_data = MyDataLoader(data_path=self.train_path, config=self.rb_config)
        self.val_data = MyDataLoader(data_path=self.val_path, config=self.rb_config)

        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        # loss function
        self.criterion = nn.NLLLoss(ignore_index=1) # ignore_index -> [PAD] token

        # log total parameters 
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def val(self, epoch):
        self.iteration(epoch, self.val_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "val"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}-{r_bar}")
        
        total_loss = 0.0
        total_iter = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            logits = self.model(data['input_ids'])

            loss = self.criterion(rearrange(logits, 'b n d -> b d n'), data['labels'].long())

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item()
            total_iter += 1
        
        mean_loss = total_loss / total_iter
        print(f"epoch: {i+1}, loss: {mean_loss:1.4f}")
 
if __name__ == "__main__":

    # trainer
    trainer = rbTrainer()

    train_loader = trainer.train(epoch=1)
    trainer.iteration(epoch=1, data_loader=train_loader)