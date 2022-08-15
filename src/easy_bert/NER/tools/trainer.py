import torch
from torch import nn as nn
import logging
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from .metrics import PRMetric
# from .metrics import conlleval


logger = logging.getLogger(__name__)



def train(epoch, model, dataloader, optimizer, scheduler, device, cfg):
    """
    training the model.
        Args:
            epoch (int): number of training steps.
            model (class): model of training.
            dataloader (dict): dict of dataset iterator. Keys are tasknames, values are corresponding dataloaders.
            optimizer (Callable): optimizer of training.
            criterion (Callable): loss criterion of training.
            device (torch.device): device of training.
            writer (class): output to tensorboard.
            cfg: configutation of training.
        Return:
            losses[-1] : the loss of training
    """
    model.train()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader):
        model.zero_grad()
        for key, value in x.items():
            x[key] = value.to(device)
        input_ids, input_mask, segment_ids = x['input_ids'], x['attention_mask'], x['token_type_ids']
        # input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model(input_ids, y, segment_ids, input_mask)
        loss = outputs
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        
        losses.append(loss.item())
        
        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
              logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
    


    return losses[-1]


def validate(args, data, model, id2label, all_ori_tokens):
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)

    logger.info("***** Running eval *****")
    
    pred_labels = []
    ori_labels = []

    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(dataloader, desc="Evaluating"):
        
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)
            

        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])
        
        for l in label_ids:
            ori_labels.append([id2label[idx.item()] for idx in l])
    
    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")
    
    # eval the model 
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    overall, by_type = conlleval.metrics(counts)
    
    return overall, by_type