import rouge_score
from rouge_score import rouge_scorer 
import numpy as np

from transformers import (
    BertTokenizer,

)

def compute_metrics(eval_preds):  
    """  
    计算评估指标  
    """   
    
    predictions, labels = eval_preds  
    # 将预测结果转换为token ID  
    predictions = np.argmax(predictions, axis=-1)  
    
    # 解码预测结果和标签  
    tokenizer:BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)  
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  
    
    # 计算ROUGE分数  
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}  
    
    for pred, label in zip(decoded_preds, decoded_labels):  
        scores = scorer.score(pred, label)  
        for key in rouge_scores:  
            rouge_scores[key].append(scores[key].fmeasure)  
    
    # 计算平均分数  
    results = {  
        "rouge1": np.mean(rouge_scores["rouge1"]),  
        "rouge2": np.mean(rouge_scores["rouge2"]),  
        "rougeL": np.mean(rouge_scores["rougeL"]),  
    }  
    
    return results  
