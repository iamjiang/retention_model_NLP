import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce
en_stopwords = set(stopwords.words('english')) 
import itertools
import re
import time

import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

def churn_date(churn_df,buffer=3,window=6):
    unum_id=[]
    policy_id=[]
    pivot_date=[]
    year=[]
    month=[]
    start_date=[]
    end_date=[]
    churn=[]
    for index,row in tqdm(churn_df.iterrows(), total=churn_df.shape[0]):
        policy_term_month=pd.to_datetime(row["policy_term_dt"]).month
        policy_anniv_month=pd.to_datetime(row["policy_anniv_dt"]).month
        if np.isnan(policy_term_month):
            date=str(row["year"])+str(int(policy_anniv_month))
            date=pd.to_datetime(str(date),format="%Y%m")
            year.append(row["year"])
            month.append(int(policy_anniv_month))
        else:
            date=str(row["year"])+str(int(policy_term_month))
            date=pd.to_datetime(str(date),format="%Y%m")
            year.append(row["year"])
            month.append(int(policy_term_month))
            
#         if int(date[4:])<10:
#             date=date[:4]+str(0)+date[4:]
        pivot_date.append(date)
        end_date.append(date-pd.offsets.DateOffset(months=buffer))
        start_date.append(date-pd.offsets.DateOffset(months=buffer+window))
        policy_id.append(row["policy_id"])
        unum_id.append(row["unum_client_id"])
        churn.append(row["churn"])
        
    churn_data=pd.DataFrame({"unum_id":unum_id,"policy_id":policy_id,"pivot_date":pivot_date,"year":year,"month":month,
                             "start_date":start_date,"end_date":end_date,"churn":churn})
    churn_data.drop_duplicates(inplace=True)
    churn_data.sort_values(by=["unum_id","policy_id","year","month"],ascending=False,inplace=True)
    churn_data['unum_id']=churn_data['unum_id'].apply(str)
    churn_data['policy_id']=churn_data['policy_id'].apply(int)
    churn_data=churn_data.reset_index(drop=True)
    return churn_data

my_folder="s3://trident-retention-output/"
folder = 's3://trident-retention-data/askunum/'

#### churn label data ####
churn_labels=pd.read_csv(my_folder+'churn_labels.csv')
churn_data=churn_date(churn_labels,buffer=3,window=6)
churn_data.to_pickle(os.path.join(my_folder,"churn_data_pickle"))

#### Askunum text data ####
askunum_text=pd.DataFrame()
for year in [2018,2019,2020,2021,2022]:
    new_data=pd.read_csv(os.path.join(my_folder,f"askunum_textbody_{year}"+".csv"))
    askunum_text=pd.concat([askunum_text,new_data])
    print("{:<15}{:<20,}".format(year,new_data.shape[0]))
    
askunum_text.drop(['Unnamed: 0'],axis=1,inplace=True)
askunum_text['unum_id']=askunum_text['unum_id'].astype(int).astype(str)
askunum_text.sort_values(["unum_id","year","month","MessageDate"],inplace=True,ascending=True)
askunum_text.to_pickle(os.path.join(my_folder,"askunum_text_pickle"))

#### Split unum_id into 10 chunk ####
def chunks_split(data,n):
    k=len(data)//n
    for i in range(0,n-1):
        yield data[i*k:(i+1)*k]
    yield data[(n-1)*k:]
    
start=time.time()
askunum_text=pd.read_pickle(os.path.join(my_folder,'askunum_text_pickle'))
end=time.time()
print("It took {:0.4f} seconds to read text data".format(end-start))

unum_id=np.unique(askunum_text.unum_id.values)
chunk=chunks_split(unum_id,10)

for i in tqdm(range(1,11)):
    tempt=pd.DataFrame(next(iter(chunk)),columns=["unum_id"]).reset_index(drop=True)
    tempt.to_csv(os.path.join(my_folder,f"unique_unum_v{i}.csv"))




