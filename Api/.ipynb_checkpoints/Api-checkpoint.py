import sys
import torch
import time

from fastapi import FastAPI
from pydantic import BaseModel

# myself
sys.path.append('/Users/sunyongdi/Desktop/nlp-code-examples')
from transformers import BertTokenizer
from src.easy_bert.TextClassifiy import models

app = FastAPI()


class Text(BaseModel):
    text: str

def init_textclassification():
    class Config:
        bert_path = 'bert-base-chinese'
        dropout = 0.3
        num_hidden = 768
        num_classes = 1
        max_len = 512

    cfg = Config()

    model_path = 'models/bert_epoch2.pth'
    model = models.BERTBaseUncased(cfg)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)
    
    return model, tokenizer, cfg

def init_ner():
    return model, tokenizer, cfg

@app.post("/api/TextClassification/bert")
async def create_item(item: Text):
    start = time.time()
    model, tokenizer, cfg = init_textclassification()
    print('加载模型时间：', time.time() - start)
    start = time.time()
    inputs = tokenizer(item.text, padding='max_length', max_length=cfg.max_len, truncation=True, return_tensors='pt')
    print('数据处理时间：', time.time() - start)
    start = time.time()
    y_pred = model(inputs)
    print('模型预测时间', time.time() - start)
    out = torch.sigmoid(y_pred).cpu().detach().numpy() >= 0.5
    return 'positive' if out.tolist()[0][0] else 'negative'


@app.post("/api/NER/bertlstmcrf")
async def create_item(item: Text):
    start = time.time()
    model, tokenizer, cfg = init_ner()
    print('加载模型时间：', time.time() - start)
    start = time.time()
    inputs = tokenizer(item.text, padding='max_length', max_length=cfg.max_len, truncation=True, return_tensors='pt')
    print('数据处理时间：', time.time() - start)
    start = time.time()
    y_pred = model(inputs)
    print('模型预测时间', time.time() - start)
    out = torch.sigmoid(y_pred).cpu().detach().numpy() >= 0.5
    return 'positive' if out.tolist()[0][0] else 'negative'


