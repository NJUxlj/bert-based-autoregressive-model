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

## This project is in progress ...

## Configuration


## Run
```bash

```



## Results
