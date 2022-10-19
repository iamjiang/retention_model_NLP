import pandas as pd
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import textwrap
import random
import time 

import spacy
nlp = spacy.load("en_core_web_md")

from transformers import AutoModelForMaskedLM , AutoTokenizer
import torch
from NLP_prompt import Prompting

import warnings
warnings.filterwarnings("ignore")

model_path="bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompting= Prompting(model=model_path)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# my_folder="s3://trident-retention-output/"

# #### Askunum text data ####
# askunum_text=pd.DataFrame()
# for year in [2018,2019,2020,2021,2022]:
#     new_data=pd.read_csv(os.path.join(my_folder,f"askunum_textbody_{year}"+".csv"))
#     askunum_text=pd.concat([askunum_text,new_data])
#     print("{:<15}{:<20,}".format(year,new_data.shape[0]))
    
# askunum_text.drop(['Unnamed: 0'],axis=1,inplace=True)
# askunum_text['unum_id']=askunum_text['unum_id'].astype(int).astype(str)
# askunum_text.sort_values(["unum_id","year","month","MessageDate"],inplace=True,ascending=True)
# askunum_text.head(2)

# start=time.time()
# df=askunum_text.groupby(["ParentId","account_id","unum_id"])['TextBody'].apply(lambda x: " ".join(x)).reset_index()
# df=df.drop_duplicates()
# end=time.time()
# print("It take {:.4f} second to group data".format(end-start))

# def truncation_text(X):
#     max_seq_length=tokenizer.model_max_length
#     truncated_input_ids=tokenizer(X,truncation=False,return_tensors="pt",add_special_tokens=False)['input_ids']
#     truncated_input_ids=truncated_input_ids[:,0:(max_seq_length - 2-2-6)].squeeze() ## 2 special tokens + 2 tokens for prefix-prompt: "email:"+ 6 tokens for post-prompt : ".this email has [MASK] sentiment"  
#     return tokenizer.decode(truncated_input_ids)

# df["truncated_TextBody"]=df["TextBody"].progress_apply(truncation_text)

# output_dir="s3://trident-retention-output/output/"
# df.to_pickle(os.path.join(output_dir,"askunum_text_truncation"))

start=time.time()
output_dir="s3://trident-retention-output/output/"
df=pd.read_pickle(os.path.join(output_dir,"askunum_text_truncation"))
end=time.time()
print("It take {:.4f} second to read data".format(end-start))

prefix_prompt="email:"
post_prompt=".this email has [MASK] sentiment"

def zero_shot_prompt(text):
    text=prefix_prompt+text+post_prompt
    prob=prompting.compute_tokens_prob(text, token_list1=["positive","neutral"], token_list2= ["negative"],device=device)[0].item()
    return prob

df1=df.sample(n=15000,random_state=202)
df1["probability"]=df1["truncated_TextBody"].progress_apply(zero_shot_prompt)
df1.sort_values("probability",ascending=True,inplace=True)

df1.to_pickle(os.path.join(output_dir,"askunum_text_sentiment_v2"))

