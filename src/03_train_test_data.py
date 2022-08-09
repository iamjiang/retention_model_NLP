import time
import os
import pickle
import pandas as pd

import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

from transformers import AutoTokenizer

my_folder="s3://trident-retention-output/"
folder = 's3://trident-retention-data/askunum/'

churn_text_pickle=pd.read_pickle(os.path.join(my_folder, "churn_text_pickle_v1"))
for i in range(2,11):
    X=pd.read_pickle(os.path.join(my_folder, f"churn_text_pickle_v{i}"))
    churn_text_pickle=pd.concat([churn_text_pickle,X])
    
churn_text_pickle=churn_text_pickle[churn_text_pickle['Full_TextBody']!='original message'] ## there are 30 observations that have textbody=='original message'


model_checkpoint="allenai/longformer-base-4096"
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

usecols=['Full_TextBody', 'Client_TextBody', 'Latest_TextBody', 'year','churn']
email_df=churn_text_pickle.loc[:,usecols]
email_df_train=email_df[email_df['year']!=2022]
email_df_test=email_df[email_df['year']==2022]

hf_dataset=Dataset.from_pandas(email_df_train)
hf_dataset=hf_dataset.remove_columns('__index_level_0__')
hf_split=hf_dataset.train_test_split(train_size=0.9, seed=101)
hf_split["validation"]=hf_split.pop("test")
hf_test=Dataset.from_pandas(email_df_test)
hf_test=hf_test.remove_columns('__index_level_0__')
hf_split["test"]=hf_test

hf_data=concatenate_datasets([hf_split["train"],  hf_split["validation"]],split="train")
hf_data=DatasetDict({"train":hf_data, "test":hf_split.pop('test')})

hf_data.save_to_disk(os.path.join(os.getcwd(),"dataset","email_all"))
email_all=load_from_disk(os.path.join(os.getcwd(),"dataset","email_all"))
email_all


