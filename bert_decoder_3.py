import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re


from transformers import BertTokenizer, BertModel

from typing import List, Tuple, Dict, Optional, Any

from config import Config, MODEL_PATH


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
        '''
        if y!=None:
            # 构建下三角矩阵, 对角线以下全为1，且包括对角线也是1
            '''
            1 0 0 0
            1 1 0 0
            1 1 1 0
            1 1 1 1
            '''
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()

            x,_ = self.bert(x, attention_mask=mask) # x.shape = (batch_size, seq_len, hidden_size)
            y_pred = self.classify(x) # shape = (batch_size, seq_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.size(2)), y.view(-1))

        else:
            #预测时，可以不使用mask
            x, _ = self.bert.forward(x) 
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)




def load_corpus(path:str="vocab.txt")->str:
    '''
    return a string, which is our corpus
    '''
    corpus = ""
    with open(file=path, mode='r', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            corpus += line

    print("load corpus successfully ~~~")
    return corpus




def build_sample(tokenizer:BertTokenizer, window_size, corpus)->Tuple[torch.LongTensor, torch.LongTensor]:
    start = random.randint(0, len(corpus)-1-window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start+1:end+1]

    x = tokenizer.encode(window)
    y = tokenizer.encode(target)

    return x,y





def build_dataset(sample_length, tokenizer, window_size, corpus)->Tuple[torch.LongTensor, torch.LongTensor]:
    '''
    sample_length: the number of samples you need
    tokenizer: 
    window_size: the fixed length of the source text
    
    '''
    dataset_x = []
    dataset_y = []

    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)



def build_model(vocab, char_dim, pretrain_model_path):
    model = BertDecoder(hidden_size=char_dim, vocab_size=len(vocab), pretrain_model_path=pretrain_model_path)
    return model



 
def generate_sentence(openings, model, tokenizer, window_size):
    '''
    Text generation test code
    '''
    pass





def sampling_strategy(prob_distribution):
    pass





def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率

    pretrain_model_path = MODEL_PATH


if __name__ == "__main__":
    load_corpus()