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
```python
def prepare_data():  
    """  
    准备CNN/DailyMail数据集，并进行必要的预处理  
    """  
    # 加载数据集，只使用一小部分进行演示  
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")  
    val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:100]")  
    
    # 初始化tokenizer  
    from transformers import BertTokenizer  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
    
    # 添加特殊token  
    special_tokens = {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}  
    tokenizer.add_special_tokens(special_tokens)  
    
    max_input_length = 512  
    max_target_length = 128  
    
    def preprocess_function(examples):  
        """  
        预处理函数：将文章和摘要转换为模型输入格式  
        """  
        # 为输入添加特殊标记  
        inputs = [f"[BOS] {text}" for text in examples['article']]  
        targets = [f"{summary} [EOS]" for summary in examples['highlights']]  
        
        # 对输入文本进行编码  
        model_inputs = tokenizer(  
            inputs,  
            max_length=max_input_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors="pt"  
        )  
        
        # 对目标摘要进行编码  
        labels = tokenizer(  
            targets,  
            max_length=max_target_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors="pt"  
        )  
        
        model_inputs["labels"] = labels["input_ids"]  
        return model_inputs  
    
    # 对数据集进行预处理  
    train_dataset = dataset.map(  
        preprocess_function,  
        batched=True,  
        remove_columns=dataset.column_names  
    )  
    
    val_dataset = val_dataset.map(  
        preprocess_function,  
        batched=True,  
        remove_columns=val_dataset.column_names  
    )  
    
    return train_dataset, val_dataset, tokenizer  
```

## This project is in progress ...

## Configuration


## Run
```bash

```



## Results
