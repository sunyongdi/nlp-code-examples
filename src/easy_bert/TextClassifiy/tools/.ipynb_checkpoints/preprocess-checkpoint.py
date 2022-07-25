
import os
import sys
import logging
from typing import List, Dict
from transformers import BertTokenizer

from ..utils import save_pkl, load_csv

logger = logging.getLogger(__name__)

def _handle_tokenizer(data: List[Dict], cfg, tokenizer):
    for d in data:
        inputs = tokenizer(
                    d['text_a'],
                    max_length=cfg.max_len,
                    padding='max_length',
                    truncation=True,
                )
        d['inputs'] = inputs


def preprocess(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)

    logger.info('convert attribution into index...')
    _handle_tokenizer(train_data, cfg, tokenizer)
    _handle_tokenizer(valid_data, cfg, tokenizer)
    _handle_tokenizer(test_data, cfg, tokenizer)

    logger.info('verify whether use pretrained language models...')
    
    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)
    logger.info('===== end preprocess data =====')
    


# if __name__ == '__main__':
#     cfg = Config()
#     cfg.cwd = os.getcwd()
#     cfg.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     cfg.data_path = '/Users/sunyongdi/Desktop/nlp-code-examples/data'
#     cfg.out_path = '/Users/sunyongdi/Desktop/nlp-code-examples/data/output'
#     cfg.max_len = 256
#     preprocess(cfg)
#     # data = load_pkl('/Users/sunyongdi/Desktop/nlp-code-examples/data/output/train.pkl')
#     # print(data)
    
    
