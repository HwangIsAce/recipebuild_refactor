import torch
from torch import nn
import tqdm

from transformers import BertConfig

from recipebuild import recipeBuild
from modelhead import rbLanguageModel
from dataloader import *
from src import bootstrap

class rbTrainer(nn.Module):
    def __init__(
            self, 
            
    ):
        super().__init__()
        # config
        self.config = BertConfig()

        self.rb_config = bootstrap.recipebuildConfig(
            path = "/home/jaesung/jaesung/research/recipebuild_retactor/config.json"
        )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data path 
        train_path = self.rb_config.processed_data_folder + '/v3_ing_title_tag/train.txt'
        val_path = self.rb_config.processed_data_folder = '/v3_ing_title_tag/val.txt'
        test_path = "" # Need to fill in

        # data loader
        self.train_data = MyDataLoader(data_path=train_path)
        # self.val_data = MyDataLoader(data_path=val_path)
        # self.test_data = MyDataLoader(data_path=test_path)

        # model        
        self.model = rbLanguageModel(recipeBuild).to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        self.criterion = nn.NLLLoss(ignore_index=1) # ignore_index -> [PAD] token

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def val(self, epoch):
        self.iteration(epoch, self.val_data, train=False)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}-{r_bar}")
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            logits = self.model(data['input_ids'])


            import IPython; IPython.embed(colors="Linux"); exit(1)
 
if __name__ == "__main__":
    trainer = rbTrainer()

    train_loader = trainer.train(epoch=1)
    trainer.iteration(epoch=1, data_loader=train_loader)