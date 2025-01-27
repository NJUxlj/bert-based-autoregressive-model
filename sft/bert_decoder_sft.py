import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

import json

from torch.utils.data import DataLoader, Dataset


from transformers import BertTokenizer, BertModel

from typing import List, Tuple, Dict, Optional, Any



import sys
sys.path.append("../")
from config.config import Config, MODEL_PATH


'''
用 Seq2Seq 的形式进行 SFT的训练


在构造 每个 x,y 样本的同时， 创建一个配套的seq2seq的 mask


'''



class BertDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super().__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)


        self.bert:BertModel= BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        # self.loss = nn.functional.cross_entropy
        # 计算loss时，忽略target中值为-1的那些位置，使他们不参与loss的计算
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
     

    def forward(self, x,  mask = None, y=None):
        '''
        x.shape = (batch_size, seq_len)
        y.shape = (batch_size, seq_len)

        return loss or y_pred
            y_pred.shape = (batch_size, seq_len, vocab_size)
        '''
        if y!=None:

            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()

            x,_ = self.bert(x, attention_mask=mask) # x.shape = (batch_size, seq_len, hidden_size)
            y_pred = self.classify(x) # shape = (batch_size, seq_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))

        else:
            x, _ = self.bert.forward(x) 
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)
        




#加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(corpus_path)->List[List[str]]:
    corpus = []
    with open(file=corpus_path, mode="r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            corpus.append([line['title'], line['content']])
            
    return corpus
    


def build_sample(tokenizer: BertTokenizer, corpus, window_size)->Tuple[torch.LongTensor]:
    '''
    return x, y, mask
    
        - x.shape = (len(s1), )
        - y.shape = (len(s2), )
        - mask.shape = (len(s1)+len(s2), len(s1)+len(s2))
    '''
    start = np.random.randint(0, len(corpus)-window_size-1)
    end = start + window_size
    
    x= corpus[start:end]
    y = corpus[start+1:end+1]

    # 为什么填 add_special_tokens=False， 因为，我们这里是language modeling, 而不是分类任务，只需要看预测的字和实际输出的字是否一致即可
    x = tokenizer.encode(x, add_special_tokens=False, padding="max_length", truncation=True, max_length = 100, return_tensors="pt")
    y = tokenizer.encode(y, add_special_tokens=False, padding="max_length", truncation=True, max_length = 100, return_tensors="pt")

    '''
    tril（输入，对角线=0，*，out=无）->张量
        返回矩阵的下三角部分（2-D张量）或一批矩阵输入，结果张量输出的其他元素设置为0。
        
        矩阵的下三角部分被定义为对角线上和对角线下的元素。
        参数对角线控制着要考虑哪条对角线。如果对角线=0，则保留主对角线上和主对角线下的所有元素。
        
        正值包括主对角线上方同样多的对角线，同样，负值排除主对角线下方同样多的斜线。
    '''

    mask_s1_s1 = torch.tril(torch.ones((len(x), len(x))))
    mask_s2_s1 = torch.ones((len(y), len(x)))
    mask_s1 = torch.cat([mask_s1_s1, mask_s2_s1], dim=0)

    mask_s1_s2 = torch.zeros((len(x), len(y)))
    mask_s2_s2 = torch.tril(torch.ones((len(y), len(y))), diagonal=0)
    mask_s2 = torch.cat([mask_s1_s2, mask_s2_s2], dim=0)

    mask = torch.cat([mask_s1, mask_s2], dim=1)

    return x, y, mask


def build_dataset(tokenizer, corpus: List[List[str]], max_length, batch_size)->DataLoader:
    '''
    return dataset_x, dataset_y, dataset_mask
        - dataset_x.shape = (batch_size, window_size)
        - dataset_y.shape = (batch_size, window_size)
        - dataset_mask.shape = (sample_length, window_size+window_size, window_size+window_size)
    '''
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        
        x:List[int] = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        # -1 代表不参与loss计算
        y:List[int] = [-1] + len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] 
        
        # 最后一个 sep_token_id 作为终止符
        
        ## 自己造
        mask = create_mask(len(prompt_encode), len(answer_encode))
        
        # padding 先截断，再补全
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        
        mask = pad_mask(mask, (max_length, max_length))

        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        
        dataset.append([x, mask, y])
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        


#构造掩码，输入两个字符串的长度
def create_mask(s1:int, s2:int)->torch.LongTensor:
    '''
    return mask.shape = (s1+s2, s1+s2)
    
    '''
    
    s1 +=2  # cls, sep
    s2 +=1 # sep
    
    mask_s1_s1 = torch.tril(torch.ones((s1, s1)))
    mask_s2_s1 = torch.ones((s2, s1))
    mask_s1 = torch.cat([mask_s1_s1, mask_s2_s1], dim=0)

    mask_s1_s2 = torch.zeros((s1, s2))
    mask_s2_s2 = torch.tril(torch.ones((s2, s2)), diagonal=0)
    mask_s2 = torch.cat([mask_s1_s2, mask_s2_s2], dim=0)

    mask = torch.cat([mask_s1, mask_s2], dim=1)
    
    return mask




# The second mask of creating a mask
def create_mask2(s1:int, s2:int)->torch.LongTensor:
    len_s1 = s1 + 2 #cls + sep
    len_s2 = s2 + 1 #sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0  
    # 遍历s2的每个token
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask 


def pad_mask(mask:torch.LongTensor, target_shape:Tuple[int, int])->torch.LongTensor:
    '''
    mask: original mask 
    
    target_shape: the desired shape of the mask, usually is (max_length, max_length)
    
    return tensor.shape = target_shape
    '''
    # original mask's hight and weight
    h = mask.shape[0]
    w = mask.shape[1]
    target_h, target_w = target_shape
    
    result = torch.zeros(target_shape, dtype=mask.dtype, device=mask.device)
    h_start = 0
    w_start = 0
    
    h_end = min(h, target_h)   
    w_end = min(w, target_w)
    
    # put the original mask into the all-zero padded mask
    result[h_start:h_end, w_start:w_end] = mask[:h_end-h_start, :w_end-w_start]
    
    return result


def build_model(vocab, char_dim, pretrain_model_path):
    return BertDecoder(char_dim, len(vocab), pretrain_model_path)



def generate_sentence(openings, model:BertDecoder, tokenizer:BertTokenizer, max_length):
    
    assert len(openings) <= max_length, "openings is too long"
    model.eval()
    with torch.no_grad():
        pre_char =  ""
        while pre_char!='\n' and len(openings)<max_length:
            # 生成下一个字
            x = tokenizer.encode(openings, add_special_tokens=False, return_tensors="pt")
            if torch.cuda.is_available():
                x = x.cuda()
                
            pred = model.forward(x.unsqueeze(0))[0][-1]
            token_id = sampling_strategy(pred)
            
            pred_char = tokenizer.decode(token_id)
            openings += pred_char
    
    return openings




def sampling_strategy(probability_distribution:torch.LongTensor):
    if random.random()>0.5:
        strategy = "greedy"
    else:
        strategy = "sampling"
        
        
    if strategy == "greedy":
        return int(torch.argmax(probability_distribution))
    else:
        return np.random.choice(list(range(len(probability_distribution))), p=probability_distribution.cpu().numpy())




def main(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    max_length = 100       #样本文本长度
    # vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = MODEL_PATH
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    
    vocab_size = len(tokenizer.vocab)   
    print("vocab_size = ", vocab_size)

    corpus = load_corpus(corpus_path)     #加载语料
    train_data:DataLoader = build_dataset(tokenizer, corpus, max_length, batch_size)  #建立数据集
    model = build_model(tokenizer.vocab, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("Text model loaded，start training")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model.forward(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\nEpoch(%d), average loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer, max_length))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer, max_length))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return








if __name__ == '__main__':
    main("./sample_sft_data.json", save_weight=False)



