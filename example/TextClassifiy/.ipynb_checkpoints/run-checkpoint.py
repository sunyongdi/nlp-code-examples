import os
import hydra
from hydra import utils

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import wandb
import logging

# import matplotlib.pyplot as plt

from src.easy_bert.TextClassifiy.tools import train, validate, CustomDataset, preprocess, collate_fn
from src.easy_bert.TextClassifiy.models.BasicBert import BERTBaseUncased
from src.easy_bert.TextClassifiy.utils.nnUtils import manual_seed

import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    cwd = utils.get_original_cwd()
    # cwd = cwd[0:-5]
    cfg.cwd = cwd
    logger.info(f'\n{cfg.pretty()}')

    wandb.init(project="test_nlp_tamplet", name=cfg.model_name)
    wandb.watch_called = False
    
    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')
    

    # 如果不修改预处理的过程，这一步最好注释掉，不用每次运行都预处理数据一次
    if cfg.preprocess:
        preprocess(cfg)
    
    train_data_path = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_data_path = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_data_path = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    
    train_dataset = CustomDataset(train_data_path)
    valid_dataset = CustomDataset(valid_data_path)
    test_dataset = CustomDataset(test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

    model = BERTBaseUncased(cfg)
    model.to(device)

    wandb.watch(model, log="all")
    logger.info(f'\n {model}')
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    cfg.dataset_len = len(train_dataset)
    num_train_steps = int(cfg.dataset_len / cfg.train_batch_size * cfg.epoch)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1, best_epoch = -1, 0
    es_loss, es_f1, es_epoch, es_patience, best_es_epoch, best_es_f1, es_path, best_es_path = 1e8, -1, 0, 0, 0, -1, '', ''
    train_losses, valid_losses = [], []
    
    if cfg.show_plot and cfg.plot_utils == 'tensorboard':
        writer = SummaryWriter('tensorboard')
    else:
        writer = None

    logger.info('=' * 10 + ' Start training ' + '=' * 10)
    
    for epoch in range(1, cfg.epoch + 1):
        manual_seed(cfg.seed + epoch)
        train_loss = train(epoch, model, train_dataloader, optimizer, scheduler, criterion, device, writer, cfg)
        valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device, cfg)
        scheduler.step(valid_loss)
        model_path = model.save(epoch, cfg)
        # logger.info(model_path)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        wandb.log({
            "train_loss":train_loss,
            "valid_loss":valid_loss
        })
    
        if best_f1 < valid_f1:
                best_f1 = valid_f1
                best_epoch = epoch
        # 使用 valid loss 做 early stopping 的判断标准
        if es_loss > valid_loss:
            es_loss = valid_loss
            es_f1 = valid_f1
            es_epoch = epoch
            es_patience = 0
            es_path = model_path
        else:
            es_patience += 1
            if es_patience >= cfg.early_stopping_patience:
                best_es_epoch = es_epoch
                best_es_f1 = es_f1
                best_es_path = es_path

    if cfg.show_plot:
        if cfg.plot_utils == 'matplot':
            plt.plot(train_losses, 'x-')
            plt.plot(valid_losses, '+-')
            plt.legend(['train', 'valid'])
            plt.title('train/valid comparison loss')
            plt.show()

        if cfg.plot_utils == 'tensorboard':
            for i in range(len(train_losses)):
                writer.add_scalars('train/valid_comparison_loss', {
                    'train': train_losses[i],
                    'valid': valid_losses[i]
                }, i)
            writer.close()

    logger.info(f'best(valid loss quota) early stopping epoch: {best_es_epoch}, '
                f'this epoch macro f1: {best_es_f1:0.4f}')
    logger.info(f'this model save path: {best_es_path}')
    logger.info(f'total {cfg.epoch} epochs, best(valid macro f1) epoch: {best_epoch}, '
                f'this epoch macro f1: {best_f1:.4f}')

    logger.info('=====end of training====')
    logger.info('')
    logger.info('=====start test performance====')
    _ , test_loss = validate(-1, model, test_dataloader, criterion, device, cfg)

    wandb.log({
        "test_loss":test_loss,
    })
    
    logger.info('=====ending====')

    
    
if __name__ == '__main__':
    main()
    