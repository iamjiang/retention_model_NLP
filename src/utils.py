import os
import time
import datetime
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import logging

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict
from datasets import load_from_disk

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
    get_scheduler
)

# from accelerate import Accelerator

# accelerator = Accelerator(fp16=True)

def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))

class Loader_Creation(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 feature_name
                ):
        super().__init__()
        self.dataset=dataset
        self.tokenizer=tokenizer
        
        self.dataset=self.dataset.map(lambda x:tokenizer(x[feature_name],truncation=True,padding="max_length"), 
                                      batched=True)
        self.dataset.set_format(type="pandas")
        self.dataset=self.dataset[:]
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,index):
        _ids = self.dataset.loc[index]["input_ids"].squeeze()
        _mask = self.dataset.loc[index]["attention_mask"].squeeze()
        _target = self.dataset.loc[index]["churn"].squeeze()
        
        return dict(
            input_ids=_ids,
            attention_mask=_mask,
            labels=_target
        )
    
    def collate_fn(self,batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask=torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        
        pad_token_id=self.tokenizer.pad_token_id
        keep_mask = input_ids.ne(pad_token_id).any(dim=0)
        
        input_ids=input_ids[:, keep_mask]
        attention_mask=attention_mask[:, keep_mask]
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
def under_sampling(df_train,target_variable, seed, negative_positive_ratio):
    np.random.seed(seed)
    LABEL=df_train[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    positive_idx=IDX[LABEL==1]
    negative_idx=np.random.choice(IDX[LABEL==0],size=(len(positive_idx)*negative_positive_ratio,))
    _idx=np.concatenate([positive_idx,negative_idx])
    under_sampling_train=df_train.loc[_idx,:]
    return under_sampling_train


def mask_creation(dataset,target_variable, seed, validation_split):
    train_idx=[]
    val_idx=[]
    
    LABEL=dataset[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    target_list=np.unique(LABEL).tolist()
        
    for i in range(len(target_list)):
        _idx=IDX[LABEL==target_list[i]]
        np.random.seed(seed)
        np.random.shuffle(_idx)
        
        split=int(np.floor(validation_split*_idx.shape[0]))
        
        val_idx.extend(_idx[ : split])
        train_idx.extend(_idx[split:])        
 
    all_idx=np.arange(LABEL.shape[0])

    val_idx=np.array(val_idx)
    train_idx=np.array(train_idx)
    
    return train_idx, val_idx


def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y.squeeze()==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight


def eval_func(data_loader,model,device,num_classes=2,loss_weight=None):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    
    model=model.to(device)
#     for batch_idx, batch in enumerate(data_loader):
    batch_idx=0
    for batch in tqdm(data_loader, position=0, leave=True):
        batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs=model(**batch)
        logits=outputs['logits']
        if loss_weight is None:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), 
                                   batch["labels"])
        else:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), 
                                   batch["labels"], weight=loss_weight.float().to(device))
            
        losses.append(loss.item())
        
        fin_targets.append(batch["labels"].cpu().detach().numpy())
        fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())   

        batch_idx+=1

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def model_evaluate(target, predicted):
    true_label_mask=[1 if (np.argmax(x)-target[i])==0 else 0 for i,x in enumerate(predicted)]
    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy=true_prediction/nb_prediction
    
    precision, recall, fscore, support = precision_recall_fscore_support(target, predicted.argmax(axis=1))
    
    try:
        auc = roc_auc_score(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())
    except:
        auc=0
        
    prec,rec,_ = precision_recall_curve(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())
    
    pr_auc=auc_score(rec,prec)
    
    arg1=predicted[:,1]
    arg2=target
    gain = lift_gain_eval(arg1,arg2,topk=[0.01,0.05,0.10,0.15,0.20])
    
    return {
        "nb_example":len(target),
        "true_prediction":true_prediction,
        "false_prediction":false_prediction,
        "accuracy":accuracy,
        "precision":precision[1], 
        "recall":recall[1], 
        "f1_score":fscore[1],
        "AUC":auc,
        "pr_auc":pr_auc,
        "GAIN":gain
    }

def lift_gain_eval(logit,label,topk):
    DF=pd.DataFrame(columns=["pred_score","actual_label"])
    DF["pred_score"]=logit
    DF["actual_label"]=label
    DF.sort_values(by="pred_score", ascending=False, inplace=True)
    gain={}
    for p in topk:
        N=math.ceil(int(DF.shape[0]*p))
        DF2=DF.nlargest(N,"pred_score",keep="first")
        gain[str(int(p*100))+"%"]=round(DF2.actual_label.sum()/(DF.actual_label.sum()*p),2)
    return gain
