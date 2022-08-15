import torch
from torch import nn as nn
import logging
from .metrics import PRMetric

logger = logging.getLogger(__name__)


def train(epoch, model, dataloader, optimizer, scheduler, criterion, device, cfg):
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

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
            
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        
        loss = criterion(y_pred, y.to(device=device, dtype=torch.int64))
        loss.backward()
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())

        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            # p r f1 皆为 macro，因为micro时三者相同，定义为acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            logger.info(f'Train Epoch {epoch}: Acc: {100. * acc:.2f}%\t'
                        f'macro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    return losses[-1]


def validate(epoch, model, dataloader, criterion, device, cfg):
    """
    validating the model.
        Args:
            epoch (int): number of validating steps.
            model (class): model of validating.
            dataloader (dict): dict of dataset iterator. Keys are tasknames, values are corresponding dataloaders.
            criterion (Callable): loss criterion of validating.
            device (torch.device): device of validating.
            cfg: configutation of validating.
        Return:
            f1 : f1 score
            loss : the loss of validating
    """
    model.eval()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x)

            loss = criterion(y_pred, y.to(device=device, dtype=torch.int64))
            # loss = criterion(y_pred, y.view(-1, 1))

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Valid Epoch {epoch}: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Test Data: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return f1, loss