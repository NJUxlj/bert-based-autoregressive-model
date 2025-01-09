'''

作业内容：实现单独的bert进行解码，生成文本

需要用到mask，自定义并传入（使用np函数+bert.forward)

改造 lstm语言模型生成文本

'''
import torch
import torch.nn as nn
from transformers import (  
    BertPreTrainedModel,   
    BertConfig,  
    Trainer,   
    TrainingArguments,  
    DataCollatorForSeq2Seq,  
) 

from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
)

from datasets import load_dataset  
from typing import Optional, Tuple, Union  
import numpy as np  
from torch.nn import CrossEntropyLoss  

from load import prepare_data
from evaluation import compute_metrics
from config import *

class BertDecoder(BertPreTrainedModel):
    def __init__(self, config: BertConfig):  
        super().__init__(config)  
        
        # 修改配置以支持因果注意力  
        config.is_decoder = True  
        self.config = config  
        
        # 初始化embeddings层  
        self.embeddings = BertEmbeddings(config)  
        
        # 初始化Transformer层  
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])  
        
        # 添加语言模型头  
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)  
        
        # 初始化权重  
        self.init_weights()  

    def get_input_embeddings(self):  
        """  
        获取输入embeddings层  
        """  
        return self.embeddings.word_embeddings  
    
    def set_input_embeddings(self, value):  
        """  
        设置输入embeddings层  
        """  
        self.embeddings.word_embeddings = value  
    
    def get_output_embeddings(self):  
        """  
        获取输出embeddings层  
        """  
        return self.lm_head  
    
    def set_output_embeddings(self, new_embeddings):  
        """  
        设置输出embeddings层  
        """  
        self.lm_head = new_embeddings
        
    def get_causal_attention_mask(self, batch_size: int, seq_length: int, dtype: torch.dtype) -> torch.Tensor:  
        """  
        生成因果注意力掩码（上三角矩阵）  
        """  
        # 创建因果掩码：确保位置i只能注意到位置j<=i  
        '''
        这句代码的功能是生成一个上三角矩阵，
        其中对角线及以下的元素为0，对角线以上的元素为1。

        具体来说：
        torch.ones((seq_length, seq_length), dtype=dtype) 创建一个形状为 (seq_length, seq_length) 的全1矩阵。
        torch.triu(..., diagonal=1) 取这个矩阵的上三角部分（包括对角线），并将对角线以下的元素设置为0

        0 1 1 1
        0 0 1 1
        0 0 0 1
        0 0 0 0
        '''
        mask = torch.triu(torch.ones((seq_length, seq_length), dtype=dtype), diagonal=1)  
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  
        return mask  
    
    def forward(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        token_type_ids: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.Tensor] = None,  
        head_mask: Optional[torch.Tensor] = None,  
        inputs_embeds: Optional[torch.Tensor] = None,  
        labels: Optional[torch.Tensor] = None,  
        output_attentions: Optional[bool] = None,  
        output_hidden_states: Optional[bool] = None,  
        return_dict: Optional[bool] = None,  
    ) -> Union[Tuple, dict]:  
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  
        
        # 获取输入的embedding表示  
        embedding_output = self.embeddings(  
            input_ids=input_ids,  
            position_ids=position_ids,  
            token_type_ids=token_type_ids,  
            inputs_embeds=inputs_embeds,  
        )  
        
        batch_size, seq_length = embedding_output.shape[:2]  
        device = embedding_output.device  
        
        # 生成因果注意力掩码  
        causal_mask = self.get_causal_attention_mask(  
            batch_size=batch_size,  
            seq_length=seq_length,  
            dtype=embedding_output.dtype,  
        ).to(device)  
        
        # 如果提供了attention_mask，则与因果掩码组合  
        if attention_mask is not None:  
            # 确保attention_mask的形状正确 [batch_size, seq_length]  
            if attention_mask.dim() == 2:  
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  
            else:
                extended_attention_mask = attention_mask
            
            # 将attention_mask与因果掩码结合
            extended_attention_mask = extended_attention_mask * (1.0 - causal_mask)  
        else:  
            extended_attention_mask = 1.0 - causal_mask  
            
        # 扩展attention_mask  
        extended_attention_mask = extended_attention_mask  
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min  
        
        # 初始化head_mask  
        if head_mask is not None:  
            if head_mask.dim() == 1:  
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)  
            elif head_mask.dim() == 2:  
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  
        else:  
            head_mask = [None] * self.config.num_hidden_layers  
            
        # 通过Transformer层  
        hidden_states = embedding_output  
        all_hidden_states = () if output_hidden_states else None  
        all_attentions = () if output_attentions else None  
        
        for i, layer_module in enumerate(self.encoder):  
            if output_hidden_states:  
                all_hidden_states = all_hidden_states + (hidden_states,)  
                
            layer_outputs = layer_module(  
                hidden_states,  
                extended_attention_mask,  
                head_mask[i],  
                output_attentions=output_attentions,  
            )  
            
            hidden_states = layer_outputs[0]  
            
            if output_attentions:  
                all_attentions = all_attentions + (layer_outputs[1],)  
                
        # 添加最后的隐藏状态  
        if output_hidden_states:  
            all_hidden_states = all_hidden_states + (hidden_states,)  
            
        # 通过语言模型头生成logits  
        logits = self.lm_head(hidden_states)  
        
        # 计算损失（如果提供了标签）  
        loss = None  
        if labels is not None:  
            # 将logits和labels展平  
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = labels[..., 1:].contiguous()  
            
            loss_fct = nn.CrossEntropyLoss()  
            loss = loss_fct(  
                shift_logits.view(-1, shift_logits.size(-1)),  
                shift_labels.view(-1)  
            )  
            
        if not return_dict:  
            output = (logits,) + (hidden_states,)  
            return ((loss,) + output) if loss is not None else output  
            
        return {  
            "loss": loss,  
            "logits": logits,  
            "hidden_states": all_hidden_states,  
            "attentions": all_attentions,  
        }  










def main():
    # 设置设备  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    # 准备数据  
    train_dataset, val_dataset, tokenizer = prepare_data()  
    
    # 创建模型配置  
    config = BertConfig.from_pretrained(MODEL_PATH)  
    config.vocab_size = len(tokenizer)  # 更新词表大小以适应新添加的特殊token  
    config.num_attention_heads = 12  # 确保这个值与预训练模型一致 
    
    # 初始化模型  
    model = BertDecoder(config)  
    model.resize_token_embeddings(len(tokenizer))  
    
    # 设置训练参数  
    training_args = TrainingArguments(  
        output_dir="./bert_decoder_summarizer",  
        num_train_epochs=3,  
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,  
        warmup_steps=500,  
        weight_decay=0.01,  
        logging_dir="./logs",  
        logging_steps=100,  
        evaluation_strategy="steps",  
        eval_steps=500,  
        save_steps=1000,  
        gradient_accumulation_steps=4,  
        fp16=True if torch.cuda.is_available() else False,  
    )  
    
    # 创建数据整理器  
    data_collator = DataCollatorForSeq2Seq(  
        tokenizer,  
        model=model,  
        label_pad_token_id=-100,  
        pad_to_multiple_of=8 if training_args.fp16 else None,  
    )  
    
    # 初始化Trainer  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=val_dataset,  
        data_collator=data_collator,  
        compute_metrics=compute_metrics,  
    )  
    
    # 开始训练  
    trainer.train()  
    
    # 保存模型  
    trainer.save_model("./bert_decoder_summarizer_final") 






if __name__ == '__main__':
    main()
