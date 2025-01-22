# Change the BERT model to a GPT-like Decoder
- 把Bert改造成自回归模型

## 基本思路
1. 使用 `CNN/DailyMail` 数据集进行 文本摘要任务 (Text-Summarization Task)
2. 修改BERT的注意力掩码机制，从双向注意力改为单向注意力（类似GPT）
3. 修改预训练任务，从MLM改为CLM（因果语言建模）
4. 调整位置编码和输入处理方式


## 主要改动
### 1. 模型架构改造：
- 继承自BertPreTrainedModel，保留了BERT的基础架构
- 设置is_decoder=True，启用因果注意力机制
- 添加了lm_head用于语言建模任务

### 2. 注意力机制改造：
- 实现了get_causal_attention_mask方法，生成上三角掩码矩阵
- 确保每个token只能看到其之前的token，实现自回归特性

### 3. 训练目标改造：
- 从BERT的MLM改为GPT式的CLM（因果语言建模）
- 实现了标准的语言模型损失计算

### 4. 前向传播流程：
- 保持了BERT的基础结构，但修改了注意力掩码的处理方式
- 添加了自回归式的预测机制

---

## Data Preprocess


## This project is in progress ...



## Evaluation Metrics
### 1. ROUGE
- ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是一个用于评估自动文本摘要和机器翻译质量的评估指标集合。它通过比较机器生成的摘要（候选摘要）与人工编写的参考摘要之间的重叠程度来进行评分。

### 2. ROUGE的主要变体
1. ROUGE-N
- 这是最基本的ROUGE度量，其中N表示我们考虑的n-gram的长度。最常用的是ROUGE-1（单个词）和ROUGE-2（双词组合）。
- ROUGE-N precision
```python
P = Count(overlapping n-grams) / Count(candidate n-grams)
# overlapping n-grams: 机器预测的摘要和金标准摘要之间，有多少个重叠的 n-grams ?
# candidate n-grams: 机器预测的摘要中一共有多少个n-grams
# 我们将机器预测出来的所有 n-grams 都视作 positives (True Positives + False Positives)
```

- ROUGE-N recall
```python
R = Count(overlapping n-grams) / Count(reference n-grams)
```

- ROUGE-N F1-Score
```python
F1 = 2 * (P * R) / (P + R)
```

3. ROUGE-L
- ROUGE-L基于最长公共子序列（LCS）计算，考虑句子级别的结构相似性。
```python
LCS(X,Y) = 最长公共子序列的长度  
ROUGE-L_recall = LCS(X,Y) / |X|  
ROUGE-L_precision = LCS(X,Y) / |Y|  
ROUGE-L_F1 = ((1 + β²) * P * R) / (R + β² * P)
```
- 其中：
    - X是参考摘要
    - Y是候选摘要
    - |X|和|Y|分别是它们的长度
    - β通常设置为1.2

5. Example



---
## Configuration
1. before running, you should manually copy the `vocab.txt` file from the bert-base-uncased directory to the this project directory.


## Run
```bash
python bert_decoder_3.py
```



## Training Snapshot


## Results
