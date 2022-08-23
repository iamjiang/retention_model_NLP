import os
import pandas as pd
pd.set_option('max_columns', 500)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce
en_stopwords = set(stopwords.words('english')) 
import itertools
import time
import re
import numpy as np
from collections import Counter

my_folder="s3://trident-retention-output/"
folder = 's3://trident-retention-data/askunum/'
if not os.path.exists(os.path.join(os.getcwd(),'outputs')):
    os.makedirs(os.path.join(os.getcwd(),'outputs'))
output_dir=os.path.join(os.getcwd(),'outputs')

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

def pipeline_askunum_counts_and_duration(folder, year, usecols=False, parent_id=False, target_columns=False): 
    if target_columns == False: 
        target_columns = ['Id', 'ParentId', 'CreatedDate', 'ClosedDate', 'account_id']
    
    def helper_get_counts_and_duration(askunum_df):
        """_summary_

        Args:
            askunum_df (pd.DataFrame): dataframe with ['Id', 'ParentId', 'account_id', 'CreatedDate', 'ClosedDate'] as columns

        Returns:
            pd.DataFrame with ['account_id', 'year', 'month', 'id_count', 'parent_id_count', 'askunum_days']
        """
        # issue counts by created date
        askunum_df['CreatedDate'] = pd.to_datetime(askunum_df['CreatedDate'])
        askunum_df['year'] = askunum_df.CreatedDate.apply(lambda x: x.year)
        askunum_df['month'] = askunum_df.CreatedDate.apply(lambda x: x.month)
        
        email_counts_by_month = askunum_df.groupby(['account_id', 'year', 'month'])[['Id']].count()
        issue_counts_by_month = askunum_df.drop('Id', axis=1).drop_duplicates().groupby(['account_id', 'year', 'month'])[['ParentId']].count()
        combined_df = email_counts_by_month.join(issue_counts_by_month)
        combined_df.rename({"Id":'askunum_id_count', 'ParentId':'askunum_parentid_count'}, axis=1, inplace=True)
        email_counts_by_month, issue_counts_by_month = None, None
        
        # completed issue durations
        askunum_df = askunum_df.loc[~askunum_df.ClosedDate.isna()]
        askunum_df['ClosedDate'] = pd.to_datetime(askunum_df['ClosedDate'])
        askunum_df['askunum_days'] = (askunum_df['ClosedDate'] - askunum_df['CreatedDate']).apply(lambda x: (x.days * 24 + x.seconds / 3600)/24)
        issue_days_by_month = askunum_df.groupby(['account_id', 'year', 'month'])[['askunum_days']].sum()
        combined_df = combined_df.join(issue_days_by_month, how='outer')
        combined_df[['askunum_id_count', 'askunum_parentid_count', 'askunum_days']].fillna(0, inplace=True)
        
        return combined_df 
    
    if year in [2018, 2019]: 
        if usecols==False: 
            usecols = ['ID', 'PARENTID', 'PARENT.CREATEDDATE', 'PARENT.CLOSEDDATE']
        if parent_id==False: 
            parent_id = 'PARENTID'
     
    if year in [2020, 2021]: 
        if usecols==False: 
            usecols = ['Id', 'ParentId', 'CreatedDate', 'ClosedDate']
        if parent_id==False: 
            parent_id = 'ParentId'
            
    if year in [2022]: 
        if usecols==False: 
            usecols = ['Id', 'ParentId', 'Parent.CreatedDate', 'Parent.ClosedDate']
        if parent_id==False: 
            parent_id = 'ParentId'
            
    askunum_df = load_askunum_df(folder, year, usecols=usecols).rename({parent_id: 'ParentId'}, axis=1) #use ParentId as the standard
    account_mapping = pd.read_csv(folder + '{}ParentAccount.csv'.format(year), usecols=['ParentId', 'Parent.AccountId']).drop_duplicates().dropna()
    askunum_df = pd.merge(askunum_df, account_mapping, on='ParentId')
    cols = ['ParentId' if i == parent_id else i for i in usecols] + ['Parent.AccountId']
    print(target_columns, cols)
    askunum_df = askunum_df.rename(dict(zip(cols, target_columns)), axis=1)
    print(askunum_df.shape)
     
    askunum_features = helper_get_counts_and_duration(askunum_df)
    askunum_features.to_csv(os.path.join(output_dir , 'askunum_issue_count_and_duration_{}.csv'.format(year)))
    print(askunum_features.shape)
    return askunum_df, askunum_features


def askunum_stemmed_bag_of_words(folder, year, nrows=None): 
    if year in [2018, 2019]: 
        id = 'ID'
        parent_id = 'PARENTID'
        text_body = 'TEXTBODY'
        created_date = 'PARENT.CREATEDDATE'      

    if year in [2020, 2021]: 
        id = 'Id'
        parent_id = 'ParentId'
        text_body = 'TextBody'
        created_date = 'CreatedDate' 

    if year in [2022]:
        id = 'Id'
        parent_id = 'ParentId'
        text_body = 'TextBody'
        created_date = 'Parent.CreatedDate'  

    askunum_text = load_askunum_df(folder, year, usecols = [id, parent_id, text_body, created_date], nrows=nrows)
    askunum_text = askunum_text.rename(dict(zip([id, parent_id, text_body, created_date], ['Id', 'ParentId', 'TextBody', 'CreatedDate'])), axis=1)

    askunum_text['TextBody'] = askunum_text['TextBody'].fillna("").astype(str).str.lower()
    askunum_text['TextBody'] = askunum_text['TextBody'].apply(lambda x: x.split('from:')[0])  # split by from:

    askunum_text['CreatedDate'] = pd.to_datetime(askunum_text['CreatedDate'])
    askunum_text['year'] = askunum_text.CreatedDate.apply(lambda x: x.year)
    askunum_text['month'] = askunum_text.CreatedDate.apply(lambda x: x.month)

    # remove phrases
    phrases = [
              'caution external email: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.', 
              'this email message and its attachments are for the sole use of the intended recipient or recipients and may contain confidential information. if you have received this email in error, please notify the sender and delete this message.',
        ]

    for p in phrases:
        askunum_text['TextBody']= askunum_text['TextBody'].str.replace(p, ' ', regex=False)

    # replace special
    for s in ['\n', '\t', '\r', '\b', '\f']:
        askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(s, ' ', regex=False)

    askunum_text['TextBody'] = askunum_text['TextBody'].str.replace('[^A-Za-z]', ' ', regex=True) # replace non-alphanumeric with space
    askunum_text['TextBody'] = askunum_text['TextBody'].str.replace(r'[ ]+', ' ', regex=True) # replace more than one space with a single space
    askunum_text['TextBodyBag'] = askunum_text['TextBody'].apply(lambda x: set(x.split(' ')) - en_stopwords) #remove stopwords
    askunum_text['TextBodyBag'] = askunum_text['TextBodyBag'].apply(lambda x: set([i for i in x if len(i)>2])) # remove short words

    my_union = lambda x: reduce(set.union, x)
    askunum_text = askunum_text.groupby(['ParentId', 'year', 'month'])['TextBodyBag'].agg(my_union) #aggregate by parent_id, year, and month

    # stem
    ps = PorterStemmer() 
    askunum_text = askunum_text.apply(lambda x: set([ps.stem(w) for w in x]))
    askunum_text.to_csv(os.path.join(output_dir ,'askunum_textbody_stemmed_bag_of_words_{}.csv'.format(year)))

    print(askunum_text.shape)
    return askunum_text


run_stemming = True
if run_stemming: 
      for year in range(2018, 2023): 
            askunum_stemmed_bag_of_words(folder, year)
            print(year, 'complete')