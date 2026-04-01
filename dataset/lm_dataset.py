from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#预训练
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length #输入给GPU的最大长度
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)
    
    #拿到jsonl的每一行，
    def __getitem__(self, index):
        sample=self.samples[index]
        # tokenizer把文本转化为input_id，
        tokens=self.tokenizer(uv 
            str(sample['text']),#假设jsonl有‘text’字段并包含在文本中
            add_special_tokens=False,
            max_length=self.max_length-2,#留出位置给BOS和EOS
            truncation=True,#如果长度超过了max，自动剪切
        ).input_ids

        #加上EOS, BOS, PAD
        tokens=[self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))
        input_ids=torch.tensor(input_ids,dtype=torch.long)#转成张量

        #编写labels，放置PAD参与loss计算
        labels=input_ids.clone()
        labels[labels==self.tokenizer.pad_token_id] = -100

        #编写mask告诉模型哪些是有效的哪些是无效的-100
        #非PAD为1，PAD为0
        attention_mask=(input_ids != self.tokenizer.pad_token_id).long()

        #输出input_ids, attention_mask, labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    