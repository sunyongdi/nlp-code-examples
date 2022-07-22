import torch.nn as nn

from transformers import BertModel

class BERTBaseUncased(nn.Module):
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