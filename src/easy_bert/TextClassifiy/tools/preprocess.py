import cgi
import os
import logging
from collections import OrderedDict
from typing import List, Dict
from transformers import BertTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import save_pkl, load_csv, load_pkl

logger = logging.getLogger(__name__)

def _handle_tokenizer(data: List[Dict], cfg, tokenizer):
    res = []
    for d in data:
        inputs = tokenizer(
                    d['text_a'],
                    max_length=cfg.max_len,
                    truncation=True,
                )
        res.append({
            'input_ids': inputs["input_ids"],
            'attention_mask': inputs["attention_mask"],
            'token_type_ids': inputs["token_type_ids"]
        })
    
    return res


def preprocess(cfg):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    # valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    # test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    # valid_data = load_csv(valid_fp)
    # test_data = load_csv(test_fp)
    # logger.info('convert attribution into index...')
    # dataset = _handle_tokenizer(train_data, cfg, tokenizer)
    # # _handle_tokenizer(valid_data)
    # # _handle_tokenizer(test_data)

    logger.info('verify whether use pretrained language models...')
    
    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    # valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    # test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    # save_pkl(valid_data, valid_save_fp)
    # save_pkl(test_data, test_save_fp)
    logger.info('===== end preprocess data =====')
    
    
class Config():
    pass

if __name__ == '__main__':
    cfg = Config()
    cfg.cwd = os.getcwd()
    cfg.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    cfg.data_path = '/Users/sunyongdi/Desktop/nlp-code-examples/data'
    cfg.out_path = '/Users/sunyongdi/Desktop/nlp-code-examples/data/output'
    cfg.max_len = 256
    preprocess(cfg)
    # data = load_pkl('/Users/sunyongdi/Desktop/nlp-code-examples/data/output/train.pkl')
    # print(data)
    
    
