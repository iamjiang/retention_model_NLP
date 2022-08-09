import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce
en_stopwords = set(stopwords.words('english')) 
import itertools
import re
import os
import pickle

import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

import textwrap

import transformers

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
)
print("Transformers version is {}".format(transformers.__version__))



my_folder="s3://trident-retention-output/"
folder = 's3://trident-retention-data/askunum/'


def load_askunum_df(folder, year, usecols=None, nrows=None): 
    if year == 2018: # ['ID', 'PARENTID', 'PARENT.CREATEDDATE', 'PARENT.CLOSEDDATE']
        askunum_df = pd.read_csv(folder + 'askunum_2018.csv', encoding='latin-1', usecols=usecols, nrows=nrows)
       
    if year == 2019: 
        askunum_df = pd.concat([pd.read_csv(folder + 'askunum_2019_{}.csv'.format(i), encoding='latin-1', usecols=usecols, nrows=nrows) for i in range(1, 4)]) 
        
    if year == 2020:  
        askunum_df = pd.concat([pd.read_csv(folder + 'unnested_2020_{}_customer.csv'.format(i), encoding='latin-1', usecols=usecols, nrows=nrows) for i in range(10)])

    if year == 2021: 
        askunum_df = pd.concat([pd.read_csv(folder + 'unnested_2021_{}_customer.csv'.format(i), encoding='latin-1', usecols=usecols, nrows=nrows) for i in range(10)]) 
        
    if year == 2022: 
        askunum_df = pd.concat([pd.read_csv(folder + 'askunum_2022_{}.csv'.format(i), encoding='latin-1', usecols=usecols, nrows=nrows) for i in range(0, 4)])
        
    return askunum_df


def askunum_textbody(folder, year, nrows=None): 
    
    if year in [2018, 2019]: 
        id = 'ID'
        parent_id = 'PARENTID'
        text_body = 'TEXTBODY'
        created_date = 'PARENT.CREATEDDATE'
        closed_date = 'PARENT.CLOSEDDATE'
        Incoming='INCOMING'
        subtype= 'PARENT.SUB_TYPE_TXT__C' 
        message_date= 'MESSAGEDATE'

    if year in [2020, 2021]: 
        id = 'Id'
        parent_id = 'ParentId'
        text_body = 'TextBody'
        created_date = 'CreatedDate' 
        closed_date = 'ClosedDate'
        Incoming='Incoming'
        subtype= 'SUB_TYPE_TXT__c'
        message_date= 'MessageDate'
            
    if year in [2022]:
        id = 'Id'
        parent_id = 'ParentId'
        text_body = 'TextBody'
        created_date = 'Parent.CreatedDate'  
        closed_date = 'Parent.ClosedDate'
        Incoming='Incoming'
        subtype= 'Parent.SUB_TYPE_TXT__c'
        message_date= 'MessageDate'

    askunum_text = load_askunum_df(folder, year, usecols = [id, parent_id, text_body, created_date, closed_date, Incoming, subtype,message_date], nrows=nrows)
    askunum_text.rename(dict(zip([id, parent_id, text_body, created_date, closed_date, Incoming, subtype,message_date], ['Id', 'ParentId', 'TextBody', \
                                                                                                                         'CreatedDate','ClosedDate','Incoming','Subtype','MessageDate'])), axis=1,inplace=True)

    askunum_text['CreatedDate'] = pd.to_datetime(askunum_text['CreatedDate'])
    askunum_text['year'] = askunum_text.CreatedDate.apply(lambda x: x.year)
    askunum_text['month'] = askunum_text.CreatedDate.apply(lambda x: x.month)

    account_mapping = pd.read_csv(folder + '{}ParentAccount.csv'.format(year), usecols=['ParentId', 'Parent.AccountId']).drop_duplicates().dropna()
    account_mapping.rename(columns={'Parent.AccountId':'account_id'},inplace=True)

    askunum_text = pd.merge(askunum_text, account_mapping, on='ParentId',how="inner")

    askunum_text["account_id"]=askunum_text["account_id"].apply(lambda x: x[:-3])

    unum_id_mapping=pd.read_csv(folder + 'retentionAskUnumcrosswalk.csv', encoding='latin-1', usecols=['Account ID', 'UNUM ID']).drop_duplicates().dropna()
    unum_id_mapping.rename(dict(zip(['Account ID', 'UNUM ID'], ['account_id','unum_id'])), axis=1, inplace=True)
    askunum_text = pd.merge(askunum_text, unum_id_mapping, on='account_id',how="inner")
    askunum_text["unum_id"]=askunum_text["unum_id"].astype(str)

    # policy_id_mapping=pd.read_csv(folder+ "PolicyData.csv", encoding='latin-1',usecols=["unum_client_id","policy_id"],nrows=None)
    # policy_id_mapping.rename({"unum_client_id":"unum_id"},axis=1,inplace=True)
    # askunum_text=pd.merge(askunum_text, policy_id_mapping, on='unum_id',how="inner")

    askunum_text.drop_duplicates(inplace=True)

    askunum_text['TextBody'] = askunum_text['TextBody'].fillna("").astype(str).str.lower()
    askunum_text['TextBody'] = askunum_text['TextBody'].apply(lambda x: x.split('from:')[0])  # split by from:

    # remove phrases
    phrases = [
              'caution external email: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.', 
              'this message originated outside of unum. use caution when opening attachments, clicking links or responding to requests for information'
              'this email message and its attachments are for the sole use of the intended recipient or recipients and may contain confidential information. if you have received this email in error, please notify the sender and delete this message.',
        ]

    for p in phrases:
        askunum_text['TextBody']= askunum_text['TextBody'].str.replace(p, ' ', regex=False)

    askunum_text['TextBody'] = askunum_text['TextBody'].str.replace('[^A-Za-z\s\.,;?]', '', regex=True) # replace non-alphanumeric with space

    # askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r"\s{2,}", " ")  # remove multiple space
    # askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r"\n{1,}", " ")  # remove multiple line breaker
    # askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r"\t{1,}", " ")  # remove multiple tab
    # askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r"(\s\.){2,}", "")  # #convert pattern really. . . . . . .  gotcha  into really. gotcha
    # askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r"[original message]", "")  # remove original message


    # replace special
    for s in ['\n', '\t', '\r', '\b', '\f']:
        askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(s, ' ', regex=False)


    askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r'[ ]+', ' ', regex=True) # replace more than one space with a single space

    def remove_long_txt(text):
        x=[i for i in text.split() if len(i)<=30] # remove long text
        x=[i for i in x if i not in ["re","original message","original message."]] # remove "original message" and "re"
        return " ".join(x)

    askunum_text['TextBody'] = askunum_text['TextBody'].apply(remove_long_txt)

    askunum_text.to_csv(my_folder + 'askunum_textbody_{}.csv'.format(year))

    return askunum_text


askunum_text_2020=askunum_textbody(folder, 2020, nrows=None)
askunum_text_2021=askunum_textbody(folder, 2021, nrows=None)
askunum_text_2022=askunum_textbody(folder, 2022, nrows=None)
askunum_text_2018=askunum_textbody(folder, 2018, nrows=None)
askunum_text_2019=askunum_textbody(folder, 2019, nrows=None)

# askunum_text=pd.concat([askunum_text_2018,askunum_text_2019,askunum_text_2020,askunum_text_2021,askunum_text_2022])
# askunum_text.drop(['Unnamed: 0'],axis=1,inplace=True)
# askunum_text['unum_id']=askunum_text['unum_id'].astype(int).astype(str)
# askunum_text.sort_values(["unum_id","year","month"],inplace=True,ascending=True)

# askunum_text.to_pickle(os.path.join(my_folder,"askunum_text_pickle"))
