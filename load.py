from datasets import load_dataset, load_from_disk
import os
from config import Config, DATASET_PATH, MODEL_PATH, PROCESSED_DATASET

from transformers import BertTokenizer  

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

def prepare_data(dataset_path = DATASET_PATH, model_path = MODEL_PATH, force_preprocess=False):  
    """  
    准备CNN/DailyMail数据集，并进行必要的预处理  
    """  

    train_data_path = os.path.join(PROCESSED_DATASET, "train")  
    val_data_path = os.path.join(PROCESSED_DATASET, "validation")  

    # 如果不强制预处理且存在已处理的数据，直接加载  
    if not force_preprocess and os.path.exists(train_data_path) and os.path.exists(val_data_path):  
        print("Loading preprocessed datasets from disk...")  
        train_dataset = load_from_disk(train_data_path)  
        val_dataset = load_from_disk(val_data_path)  
        
        # 加载tokenizer  
        tokenizer = BertTokenizer.from_pretrained(model_path)  
        special_tokens = {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}  
        tokenizer.add_special_tokens(special_tokens)  
        
        print("Datasets loaded successfully!")  
        return train_dataset, val_dataset, tokenizer  
    
    print("Preprocessing datasets...") 


    # 加载数据集，只使用一小部分进行演示  
    train_dataset = load_dataset(dataset_path, "3.0.0")['train'].select(range(1000))
    val_dataset = load_dataset(dataset_path, "3.0.0")['validation'].select(range(100))
    
    # 初始化tokenizer  
    tokenizer = BertTokenizer.from_pretrained(model_path)  
    
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
    # 创建保存目录（如果不存在）  
    os.makedirs(PROCESSED_DATASET, exist_ok=True)  

    # 保存处理后的数据集到本地  
    print("Saving preprocessed datasets to disk...")  
    
    train_dataset.save_to_disk(train_data_path)  
    val_dataset.save_to_disk(val_data_path)  
    print("Datasets saved successfully!") 

    return train_dataset, val_dataset, tokenizer  
