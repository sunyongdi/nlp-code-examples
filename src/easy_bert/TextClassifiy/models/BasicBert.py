import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch.nn as nn

from BasicModule import BasicModule

from transformers import BertModel

class BERTBaseUncased(BasicModule):
    def __init__(self, cfg):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(cfg.bert_path)
        self.bert_drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.num_hidden, cfg.num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input = """I am sunyd!"""
    inputs = tokenizer(input, max_length=256, padding='max_length', return_tensors="pt")

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    bert  = BERTBaseUncased()
    output = bert(ids, mask, token_type_ids)
    print(output)
    