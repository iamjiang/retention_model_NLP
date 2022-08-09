import os
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import missingno as msno
import pickle
import lightgbm
import xgboost as xgb
#tuning hyperparameters
from bayes_opt import BayesianOptimization
from skopt  import BayesSearchCV 

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import auc as auc_score

import warnings
warnings.filterwarnings('ignore')

data_dir=os.getcwd()
policy_premium_df = pd.read_csv(os.path.join(data_dir,'PolicyPremium_Combined.csv'))

data_dir="/app/models/trident/retention/engineered_update"
churn_labels = pd.read_csv(os.path.join(data_dir,'churn_labels.csv'))
churn_labels.dropna(subset=['churn'],inplace=True)

policy_id=np.unique(churn_labels['policy_id'].values)
policy_premium_df=policy_premium_df[policy_premium_df["policy_id"].isin(policy_id)]

policy_premium_df['year']  = policy_premium_df.bill_due_dt.apply(lambda x: x[:4])
policy_premium_df['month'] = policy_premium_df.bill_due_dt.apply(lambda x: x[5:7])
policy_premium_df['policy_id']=policy_premium_df['policy_id'].astype(int)
policy_premium_df['year']=policy_premium_df['year'].apply(str)
policy_premium_df['month']=policy_premium_df['month'].apply(str)
policy_premium_df.drop(['Unnamed: 0','report_start_dt','report_end_dt','bill_due_dt','bill_gen_dt'], axis=1, inplace=True)
policy_premium_df=policy_premium_df.reset_index(drop=True)

policy_premium_df = policy_premium_df.replace('?', np.nan)
policy_premium_df["AvgPdBilldueDays"]=policy_premium_df["AvgPdBilldueDays"].astype(float)
policy_premium_df["AvgPdBillLstGenDays"]=policy_premium_df["AvgPdBillLstGenDays"].astype(float)
policy_premium_df["Lag12_cntBillGens"]=policy_premium_df["Lag12_cntBillGens"].astype(float)
policy_premium_df["Lag12_cntPaidFull"]=policy_premium_df["Lag12_cntPaidFull"].astype(float)
policy_premium_df["Lag12_cntFirstGenPaidFull"]=policy_premium_df["Lag12_cntFirstGenPaidFull"].astype(float)
policy_premium_df["Lag12_cntBills"]=policy_premium_df["Lag12_cntBills"].astype(float)

policy_premium_df['CountBills'] = policy_premium_df['CountBills'].replace(0, np.nan)
policy_premium_df['AvgPdBilldueDays'] = policy_premium_df['AvgPdBilldueDays'].replace(0, np.nan)
policy_premium_df['AvgPdBillLstGenDays'] = policy_premium_df['AvgPdBillLstGenDays'].replace(0, np.nan)
policy_premium_df['AvgBillGenCnt'] = policy_premium_df['AvgBillGenCnt'].replace(0, np.nan)
policy_premium_df['AvgPaidFullCnt'] = policy_premium_df['AvgPaidFullCnt'].replace(0, np.nan)


### Fill missing values by simple imputation
my_imputer = SimpleImputer()
numeric_columns=[]
categorical_columns=[]
for c in policy_premium_df.columns:
    if policy_premium_df[c].dtypes!="object":
        numeric_columns.append(c)
    else:
        categorical_columns.append(c)

data_numeric=policy_premium_df.loc[:,numeric_columns]
data_numeric=pd.DataFrame(my_imputer.fit_transform(data_numeric),columns=numeric_columns)

data_categorical=policy_premium_df.loc[:,categorical_columns]
policy_premium_df = pd.concat([data_numeric, data_categorical], axis = 1)

policy_premium_df['policy_id']=policy_premium_df['policy_id'].astype(int)
policy_premium_df['year']=policy_premium_df['year'].apply(int)
policy_premium_df['month']=policy_premium_df['month'].apply(int)


def policy_df_accumulate(df):
    df.sort_values(['policy_id','year','month'],inplace=True)
    df["paid_bill_prop"]=df['CurrPaidAmt'].astype(float)/df['CurrBillAmt'].astype(float)
    # df["idx"]=df.groupby(['policy_id',"year"]).ngroup()
    exc_col=["policy_id","Lag12_cntBillGens","Lag12_cntPaidFull","Lag12_cntFirstGenPaidFull","Lag12_cntBills","year","month"]
    for col in tqdm(df.columns):
        if col not in exc_col:
            if col not in ["OrigBillAmt","CurrBillAmt","CurrPaidAmt","PaidBillDueDays","AvgPdBilldueDays","PaidBillLastGenDays","AvgPdBillLstGenDays","paid_bill_prop"]:
                df["L12_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(12, min_periods=1).sum()))
                df["L6_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(6, min_periods=1).sum()))
                df["L1_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(1, min_periods=1).sum()))
                df["L2_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(2, min_periods=1).sum()))
                df["L3_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(3, min_periods=1).sum()))
            else:
                df["L12_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(12, min_periods=1).mean()))
                df["L6_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(6, min_periods=1).mean()))
                df["L1_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(1, min_periods=1).mean()))
                df["L2_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(2, min_periods=1).mean()))
                df["L3_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(3, min_periods=1).mean()))  
                
#                 df["std12_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(12, min_periods=1).std()))
#                 df["std6_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(6, min_periods=1).std()))
#                 df["std1_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(1, min_periods=1).std()))
#                 df["std2_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(2, min_periods=1).std()))
#                 df["std3_"+col]=(df.groupby(["policy_id"])[col].apply(lambda x: x.rolling(3, min_periods=1).std()))
                
        if col not in ["policy_id","year","month"]:
            df["lag1_"+col]=df[col].shift(1)
            df["d1_"+col]=df[col]-df["lag1_"+col]
            df["r1_"+col]=(df[col]-df["lag1_"+col])/df["lag1_"+col]
            df.drop(["lag1_"+col],axis=1,inplace=True)
            
            df["lag2_"+col]=df[col].shift(2)
            df["d2_"+col]=df[col]-df["lag2_"+col]
            df["r2_"+col]=(df[col]-df["lag2_"+col])/df["lag2_"+col]
            df.drop(["lag2_"+col],axis=1,inplace=True)

            df["lag3_"+col]=df[col].shift(3)
            df["d3_"+col]=df[col]-df["lag3_"+col]
            df["r3_"+col]=(df[col]-df["lag3_"+col])/df["lag3_"+col]
            df.drop(["lag3_"+col],axis=1,inplace=True)
            
            df["lag6_"+col]=df[col].shift(6)
            df["d6_"+col]=df[col]-df["lag6_"+col]
            df["r6_"+col]=(df[col]-df["lag6_"+col])/df["lag6_"+col]
            df.drop(["lag6_"+col],axis=1,inplace=True)
            
            df["lag12_"+col]=df[col].shift(12)
            df["d12_"+col]=df[col]-df["lag12_"+col]
            df["r12_"+col]=(df[col]-df["lag12_"+col])/df["lag12_"+col]
            df.drop(["lag12_"+col],axis=1,inplace=True)
            
    # df.drop(['idx'],axis=1,inplace=True) 
               
    return df

df=policy_df_accumulate(policy_premium_df)

file_output="policy_premium_pickle"
data_dir="/app/models/dij22"
df.to_pickle(os.path.join(data_dir,file_output))

