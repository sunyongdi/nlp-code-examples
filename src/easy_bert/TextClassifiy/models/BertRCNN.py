# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer

from .BasicModule import BasicModule





class BertRCNN(BasicModule):

    def __init__(self, config):
        super(BertRCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size, stride=256)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
        # self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        encoder_out, text_cls = self.bert(**x, return_dict=False)
        out, (hidden_last,cn_last) = self.lstm(encoder_out)
        
        # hidden_last_L=hidden_last[-2]
        # #print(hidden_last_L.shape)  #[32, 384]
        # #反向最后一层，最后一个时刻
        # hidden_last_R=hidden_last[-1]
        # #print(hidden_last_R.shape)   #[32, 384]
        # #进行拼接
        # out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        # out = self.dropout(out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out