import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re


from transformers import BertTokenizer, BertModel




class BertDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super().__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)


        self.bert = BertModel.from_pretrained()
        self.classify = nn.Linear()
        self.loss = nn.functional.cross_entropy
    

    def forward(self):
        pass




def load_corpus(path):
    pass




def build_sample(tokenizer, window_size, corpus):
    pass




def build_dataset(sample_length, tokenizer, window_size, corpus):
    pass



def build_model(vocab, char_dim, pretrain_model_path):
    pass




 
def generate_sentence(openings, model, tokenizer, window_size):
    pass





def sampling_strategy(prob_distribution):
    pass





def train(corpus_path, save_weight=True):
    pass




if __name__ == "__main__":
    pass