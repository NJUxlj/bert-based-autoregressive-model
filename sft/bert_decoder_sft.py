import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re


from transformers import BertTokenizer, BertModel

from typing import List, Tuple, Dict, Optional, Any

from ..config.config import Config, MODEL_PATH




class BertDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super().__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)


        self.bert:BertModel= BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
    

    def forward(self, x, y=None):
        '''
        x.shape = (batch_size, seq_len)
        y.shape = (batch_size, seq_len)

        return loss or y_pred
            y_pred.shape = (batch_size, seq_len, vocab_size)
        '''
        if y!=None:

            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()

            x,_ = self.bert(x, attention_mask=mask) # x.shape = (batch_size, seq_len, hidden_size)
            y_pred = self.classify(x) # shape = (batch_size, seq_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))

        else:
            x, _ = self.bert.forward(x) 
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)