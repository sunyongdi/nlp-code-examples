# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer

from .BasicModule import BasicModule





class BertRNN(BasicModule):

    def __init__(self, config):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        encoder_out, _ = self.bert(**x, return_dict=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out