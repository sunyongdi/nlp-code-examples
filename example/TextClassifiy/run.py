import os
import hydra
import torch
import logging
from hydra import utils
import warnings
warnings.filterwarnings('ignore')
import sys
from src.easy_bert.TextClassifiy.tools.preprocess import preprocess
from src.easy_bert.TextClassifiy.tools.dataset import CustomDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader


def collate_fn_s(batch_size):
    inputs = []
    labels = []
    for i, j in batch_size:
        inputs.append(i)
        labels.append(j)
    return inputs, labels
# import wandb
@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    # cfg.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if cfg.preprocess:
        preprocess(cfg)
    train_data_path = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    train_dataset = CustomDataset(train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    for inputs, lables in train_dataloader:
        print('inputs', inputs)
        tokenlist = tokenizer(
                    inputs,
                    max_length=cfg.max_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
        out = model(**tokenlist)
        print(out)
        break
    
if __name__ == '__main__':
    main()
    