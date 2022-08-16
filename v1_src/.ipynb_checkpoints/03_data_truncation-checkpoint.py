import os
import numpy as np
import pandas as pd
import argparse


from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

import transformers
print("Transformers version is {}".format(transformers.__version__))

from transformers import AutoTokenizer

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Data_Truncation')
    parser.add_argument("--truncation_strategy", type=str, default="tail",help="how to truncate the long length email")
    parser.add_argument('--model_checkpoint', type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--model_name", default="longformer", type=str)
    parser.add_argument("--max_length", type=int, default=2000,help="maximal input length")
    parser.add_argument("--feature_name", default="Full_TextBody", type=str)
    
    args = parser.parse_args()
    
    print(args)
    
data_dir=os.path.join(os.getcwd(),"dataset","email_all")
email_all=load_from_disk(data_dir)
email_all=email_all.filter(lambda x: x[args.feature_name]!=None)

tokenizer=AutoTokenizer.from_pretrained(args.model_checkpoint)

max_seq_length=args.max_length
def truncation_text(example):
    truncated_input_ids=tokenizer(example[args.feature_name],truncation=True,padding=False,return_tensors="pt",add_special_tokens=False)['input_ids']

    if args.truncation_strategy=="tail":
        truncated_input_ids=truncated_input_ids[:,-(max_seq_length - 2):].squeeze()
    elif args.truncation_strategy=="head":
        truncated_input_ids=truncated_input_ids[:,0:(max_seq_length - 2)].squeeze()
    elif args.truncation_strategy=="mixed":
        truncated_input_ids=truncated_input_ids[:(max_seq_length - 2) // 2] + truncated_input_ids[-((max_seq_length - 2) // 2):]
        truncated_input_ids=truncated_input_ids.squeeze()
    else:
        raise NotImplemented("Unknown truncation. Supported truncation: tail, head, mixed truncation")

    return {"truncated_text":tokenizer.decode(truncated_input_ids)}

email_all=email_all.map(truncation_text)
columns=email_all['train'].column_names
columns_to_keep=['truncated_text','churn']
columns_to_remove=set(columns)-set(columns_to_keep)
email_all=email_all.remove_columns(columns_to_remove)
email_all=email_all.rename_column("truncated_text", args.feature_name)    

email_all.save_to_disk(os.path.join(os.getcwd(),"dataset",args.feature_name+"_truncation_"+args.truncation_strategy+"_"+args.model_name))

