import argparse
import pandas as pd
import numpy as np
from numpy import savez_compressed, load
import re
import time
import os
import pickle
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
from collections import Counter
import textwrap

def dataframe_to_dictionary(df, key_column): 

    dict_of_dataframes = dict()
    df[key_column]=df[key_column].astype(int).apply(str)
    df=df.dropna(subset=["TextBody"])
    df["year_month"]=pd.to_datetime(df.apply(lambda x: str(x['year'])+ str(x['month']) ,axis=1),format="%Y%m")
      
    counter = 0 
    for id, group in tqdm(df.groupby(key_column),total=df.unum_id.unique().shape[0], position=0, leave=True):
        group = group.sort_values("year_month")
        group = group.set_index("year_month")
        dict_of_dataframes[id] = group
        counter += 1
    return dict_of_dataframes

def main(args,churn_df, dict_of_df,  num_months_to_use=6):
    
    churn_df["unum_id"]=churn_df["unum_id"].apply(str)
    
    unum_id=[]
    policy_id=[]
    pivot_date=[]
    year=[]
    month=[]
    start_date=[]
    end_date=[]
    Full_TextBody=[]
    Client_TextBody=[]
    Latest_TextBody=[]
    EMAIL_COUNT=[]
    ISSUE_COUNT=[]
    DURATION=[]
    SUBTYPE=[]
    churn=[]

    for index,row in tqdm(churn_df.iterrows(), total=churn_df.shape[0]):
        id=row["unum_id"]

        if id not in dict_of_df:
            continue
        data = dict_of_df[id]
        tempt=data[(data.index>=row["start_date"]) & (data.index<=row["end_date"])]
        
        if tempt.empty:
            continue
            
        tempt=tempt.reset_index()
        tempt.dropna(subset=['TextBody'],inplace=True)
        tempt=tempt[tempt.TextBody.notna()]
        tempt=tempt[tempt.TextBody.str.len()>0]
        
        tempt.sort_values(["unum_id","MessageDate"],inplace=True,ascending=True)
        
        subtype=Counter(tempt[tempt["Incoming"]==True].Subtype.values)
        email_counts = tempt[tempt["Incoming"]==True].shape[0]
        issue_counts = tempt["ParentId"].unique().shape[0]
        tempt['askunum_days'] = (pd.to_datetime(tempt['ClosedDate']) - pd.to_datetime(tempt['CreatedDate'])).apply(lambda x: (x.days * 24 + x.seconds / 3600)/24)
        tempt2=tempt.drop_duplicates(subset=["ParentId"])
        duration=np.sum(tempt2['askunum_days'].values)

        tempt_1=tempt.groupby(["unum_id"])["TextBody"].apply(lambda x : ".".join(x)).reset_index()
        Full_TextBody.append(tempt_1["TextBody"][0])

        tempt_2=tempt[tempt["Incoming"]==True]
        tempt_2=tempt_2.groupby(["unum_id"])["TextBody"].apply(lambda x : ".".join(x)).reset_index()
        if tempt_2.empty:
            Client_TextBody.append(None)
        else:
            Client_TextBody.append(tempt_2["TextBody"][0])

        tempt_3=tempt.sort_values(["unum_id","ParentId","MessageDate"],ascending=True)
        tempt_3.drop_duplicates(subset=["unum_id","ParentId"],keep="last",inplace=True)
        tempt_3=tempt_3.groupby(["unum_id"])["TextBody"].apply(lambda x : ".".join(x)).reset_index()
        if tempt_3.empty:
            Latest_TextBody.append(None)
        else:
            Latest_TextBody.append(tempt_3["TextBody"][0])        

        EMAIL_COUNT.append(email_counts)
        ISSUE_COUNT.append(issue_counts)
        DURATION.append(duration)
        SUBTYPE.append(subtype)
    
        unum_id.append(row["unum_id"])
        policy_id.append(row["policy_id"])
        pivot_date.append(row["pivot_date"])
        year.append(row["year"])
        month.append(row["month"])
        start_date.append(row["start_date"])
        end_date.append(row["end_date"])
        churn.append(row["churn"])
        
    churn_text_data=pd.DataFrame({"unum_id":unum_id,"policy_id":policy_id,"pivot_date":pivot_date,"year":year,"month":month,\
                                 "start_date":start_date,"end_date":end_date,"Full_TextBody":Full_TextBody,"Client_TextBody":Client_TextBody,\
                                  "Latest_TextBody":Latest_TextBody,"email_counts":EMAIL_COUNT,"issue_counts":ISSUE_COUNT,"duration":DURATION,"subtype":SUBTYPE,"churn":churn})
    
    ## drop data if Client email is None
    churn_text_data.dropna(subset=["Client_TextBody"],inplace=True)
    
    # churn_text_data.drop_duplicates(inplace=True)
    churn_text_data.sort_values(by=["unum_id","policy_id","year","month"],ascending=False,inplace=True)
    churn_text_data['unum_id']=churn_text_data['unum_id'].apply(str)
    churn_text_data['policy_id']=churn_text_data['policy_id'].apply(int)
    churn_text_data=churn_text_data.reset_index(drop=True)
    
    print("{:<20}{:<20,}".format("rows of data:",churn_text_data.shape[0]))
    
    churn_text_data.to_pickle(os.path.join(my_folder,args.output_name))
    
    # return churn_text_data
    
    
if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("create churn text dataset")

    argparser.add_argument('--output_name', type=str, default="churn_text_pickle") 
    
    args = argparser.parse_args()
    
    print(args)
    
    my_folder="s3://trident-retention-output/"
    folder = 's3://trident-retention-data/askunum/'
    
    start=time.time()
    askunum_text=pd.read_pickle(os.path.join(my_folder,'askunum_text_pickle'))
    end=time.time()
    print("It took {:0.4f} seconds to read text data".format(end-start))
    
    churn_data=pd.read_pickle(os.path.join(my_folder,'churn_data_pickle'))
    
    df_of_dict = dataframe_to_dictionary(askunum_text, 'unum_id')

    main(args,churn_data, df_of_dict,  num_months_to_use=6)
    

# import textwrap
# import random

# # Wrap text to 80 characters.
# wrapper = textwrap.TextWrapper(width=120) 

# exam_1 = test["Full_TextBody"]
# exam_2 = test["Client_TextBody"]
# exam_3 = test["Latest_TextBody"]

# # Randomly choose some examples.
# for i in range(3):
#     random.seed(101+i)
#     j = random.choice(exam_1.index)
    
#     print('')
#     print(wrapper.fill(exam_1[j]))
#     print('')
#     print(wrapper.fill(exam_2[j]))
#     print('')
#     print(wrapper.fill(exam_3[j]))
#     print('')
#     print("*"*50)