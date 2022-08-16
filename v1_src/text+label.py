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

import textwrap

def main(args,churn_df,text_df):
    
    churn_df["unum_id"]=churn_df["unum_id"].apply(str)
    text_df["unum_id"]=text_df["unum_id"].astype(int).apply(str)
    
    file_name=args.unum_id
    unique_unum_id=pd.read_csv(os.path.join(my_folder,file_name+".csv"), usecols=["unum_id"])["unum_id"].values.squeeze()
    unique_unum_id=unique_unum_id.astype(str)
    
    text_df=text_df[text_df['unum_id'].isin(unique_unum_id)]
    churn_df=churn_df[churn_df["unum_id"].isin(unique_unum_id)]
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
    Subtype=[]
    churn=[]

    # text_df.loc[:,'day']=1
    # text_df.loc[:,'date']=pd.to_datetime(text_df[['year','month','day']],format="%Y%m%d")
    # text_df.drop(['day'],inplace=True, axis=1)

    for index,row in tqdm(churn_df.iterrows(), total=churn_df.shape[0]):

        ### Concatenate email message between start_date and end_date
        tempt=text_df[(text_df["unum_id"]==row["unum_id"]) & (pd.to_datetime(text_df["MessageDate"]).dt.date>=row["start_date"]) &  (pd.to_datetime(text_df["MessageDate"]).dt.date<=row["end_date"])]
        tempt.dropna(subset=["TextBody"],inplace=True)

        if tempt.empty:
            continue

        tempt.sort_values(["unum_id","MessageDate"],inplace=True,ascending=True)
        tempt2=tempt.drop_duplicates(subset=["unum_id"],keep="last")
        Subtype.append(tempt.drop_duplicates(subset=["unum_id"],keep="last")["Subtype"].values[0])

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
                                  "Latest_TextBody":Latest_TextBody,"Subtype":Subtype,"churn":churn})
    
    # churn_text_data.drop_duplicates(inplace=True)
    churn_text_data.sort_values(by=["unum_id","policy_id","year","month"],ascending=False,inplace=True)
    churn_text_data['unum_id']=churn_text_data['unum_id'].apply(str)
    churn_text_data['policy_id']=churn_text_data['policy_id'].apply(int)
    churn_text_data=churn_text_data.reset_index(drop=True)
    
    churn_text_data.to_pickle(os.path.join(my_folder,args.output_name))
    
    # return churn_text_data
    
    
if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("create churn text dataset")
    
    argparser.add_argument('--unum_id', type=str, default="unique_unum_v1")

    argparser.add_argument('--output_name', type=str, default="churn_text_pickle_v1") 
    
    args = argparser.parse_args()
    
    print(args)
    
    my_folder="s3://trident-retention-output/"
    folder = 's3://trident-retention-data/askunum/'
    
    start=time.time()
    askunum_text=pd.read_pickle(os.path.join(my_folder,'askunum_text_pickle'))
    end=time.time()
    print("It took {:0.4f} seconds to read text data".format(end-start))
    
    churn_data=pd.read_pickle(os.path.join(my_folder,'churn_data_pickle'))

    main(args,churn_data,askunum_text)
    
    
    

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