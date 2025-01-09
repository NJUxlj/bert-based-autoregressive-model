from datasets import load_dataset
import os

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

def prepare_data():  
    """  
    准备CNN/DailyMail数据集，并进行必要的预处理  
    """  
    # 加载数据集，只使用一小部分进行演示  
    train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")  
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
    train_dataset = train_dataset.map(  
        preprocess_function,  
        batched=True,  
        remove_columns=train_dataset.column_names  
    )  
    
    val_dataset = val_dataset.map(  
        preprocess_function,  
        batched=True,  
        remove_columns=val_dataset.column_names  
    )  
    
    return train_dataset, val_dataset, tokenizer  
