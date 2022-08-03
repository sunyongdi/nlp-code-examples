# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModule import BasicModule

from transformers import BertModel

from . import BasicModule

class BertCNN(BasicModule):

    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * 3, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder_out, text_cls = self.bert(**x, return_dict=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out