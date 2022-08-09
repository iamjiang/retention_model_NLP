import os
# os.chdir(r"/content/drive/MyDrive/billing_features/raw/")
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
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

file="policy_premium_pickle"
data_dir="/app/models/dij22"
policy_premium_df=pd.read_pickle(os.path.join(data_dir,file))

data_dir="/app/models/trident/retention/engineered_update_2022_06"
churn_labels = pd.read_csv(os.path.join(data_dir,'churn_labels.csv'))
churn_labels.dropna(subset=['churn'],inplace=True)


# positive_sample=churn_labels[churn_labels.churn==1]
# check1=0
# check2=0
# for index,row in tqdm(positive_sample.iterrows(), total=positive_sample.shape[0]):
    
#     if pd.to_datetime(row["policy_anniv_dt"])==pd.to_datetime(row["policy_term_dt"]):
#         check1+=1
        
#     if np.isnan(pd.to_datetime(row["policy_term_dt"]).month):
#         check2+=1
        
        

# print(f"Among all churn policy {positive_sample.shape[0]:,}, there are {check1/positive_sample.shape[0]:.2%} data whose policy_anniv_dt==policy_term_dt")
# print(f"There are {check2:,} that churn but policy_term_dt is missing")



def data_merge_yearly(churn_df, policy_df, buffer):
    policy_id=[]
    policy_anniv_dt=[]
    orig_policy_eff_dt=[]
    policy_term_dt=[]
    year=[]
    month=[]
    pivot_date=[]
    survival_month=[]
    churn=[]
    for index,row in tqdm(churn_df.iterrows(), total=churn_df.shape[0]):           
            
        policy_term_month=pd.to_datetime(row["policy_term_dt"]).month
        policy_anniv_month=pd.to_datetime(row["policy_anniv_dt"]).month
        if np.isnan(policy_term_month):
            date1=str(row["year"])+str(int(policy_anniv_month))
            date2=pd.to_datetime(str(date1),format="%Y%m")
        else:
            date1=str(row["year"])+str(int(policy_term_month))
            date2=pd.to_datetime(str(date1),format="%Y%m")-pd.offsets.DateOffset(months=buffer)

        policy_id.append(row["policy_id"])
        year.append(date2.year)
        month.append(date2.month)
        
        if int(date1[4:])<10:
            date1=date1[:4]+str(0)+date1[4:]
        pivot_date.append(date1)
        
        
        x2=pd.to_datetime(row["target_dt"])
        x1=pd.to_datetime(row["orig_policy_eff_dt"])
        survival_month.append(int((x2-x1)/np.timedelta64(1,"M")))
        
        policy_anniv_dt.append(row["policy_anniv_dt"])
        policy_term_dt.append(row["policy_term_dt"])
        orig_policy_eff_dt.append(row["orig_policy_eff_dt"])
        
        churn.append(row["churn"])
        
    churn_data=pd.DataFrame({"policy_id":policy_id,"orig_policy_eff_dt":orig_policy_eff_dt,"policy_anniv_dt":policy_anniv_dt,"policy_term_dt":policy_term_dt,
                             "pivot_date":pivot_date,"year":year,"month":month,"survival_month":survival_month,"churn":churn})
    # churn_data["month"]=output["month"].apply(lambda x: str(x) if x>=10 else str(0)+str(x))
    churn_data.drop_duplicates(inplace=True)
    churn_data['policy_id']=churn_data['policy_id'].astype(int)
    churn_data['year']=churn_data['year'].apply(int)
    churn_data['month']=churn_data['month'].apply(int)
    
    df=pd.merge(churn_data,policy_df,how="inner", on=["policy_id", "year","month"])
    df.sort_values(by=["policy_id","year"],ascending=True,inplace=True)
    
    return df


def data_merge_monthly(churn_df, policy_df, fixed_window, buffer):
    policy_id=[]
    policy_anniv_dt=[]
    orig_policy_eff_dt=[]
    policy_term_dt=[]
    year=[]
    month=[]
    pivot_date=[]
    survival_month=[]
    churn=[]
    for index,row in tqdm(churn_df.iterrows(), total=churn_df.shape[0]):
        policy_term_month=pd.to_datetime(row["policy_term_dt"]).month
        policy_anniv_month=pd.to_datetime(row["policy_anniv_dt"]).month
        if np.isnan(policy_term_month):
            date1=str(row["year"])+str(int(policy_anniv_month))
            for t in range(0, fixed_window+0+2*fixed_window,3): ## if not churn, don't need buffer
                date2=pd.to_datetime(str(date1),format="%Y%m")-pd.offsets.DateOffset(months=t)
                policy_id.append(row["policy_id"])
                year.append(date2.year)
                month.append(date2.month)
            
                # if int(date1[4:])<10:
                #     date1=date1[:4]+str(0)+date1[4:]
                pivot_date.append(date1)

                x2=pd.to_datetime(date2)
                x1=pd.to_datetime(row["orig_policy_eff_dt"])
                survival_month.append(int((x2-x1)/np.timedelta64(1,"M")))
                
                policy_anniv_dt.append(row["policy_anniv_dt"])
                policy_term_dt.append(row["policy_term_dt"])
                orig_policy_eff_dt.append(row["orig_policy_eff_dt"])
                
                churn.append(0)
            
            
        else:
            date1=str(row["year"])+str(int(policy_term_month))
        
            for t in range(buffer, fixed_window+buffer+2*fixed_window,3): ## if churn, need buffer

                date2=pd.to_datetime(str(date1),format="%Y%m")-pd.offsets.DateOffset(months=t)
                policy_id.append(row["policy_id"])
                year.append(date2.year)
                month.append(date2.month)
            
                # if int(date1[4:])<10:
                #     date1=date1[:4]+str(0)+date1[4:]
                pivot_date.append(date1)

                x2=pd.to_datetime(date2)
                x1=pd.to_datetime(row["orig_policy_eff_dt"])
                survival_month.append(int((x2-x1)/np.timedelta64(1,"M")))
                
                policy_anniv_dt.append(row["policy_anniv_dt"])
                policy_term_dt.append(row["policy_term_dt"])
                orig_policy_eff_dt.append(row["orig_policy_eff_dt"])

                if date2==pd.to_datetime(str(date1),format="%Y%m")-pd.offsets.DateOffset(months=buffer) and row["churn"]==1:
                    churn.append(1)
                else:
                    churn.append(0)
                
                
        # ## Add the decision point on anniversary date
        # if (pd.to_datetime(row["policy_anniv_dt"]).year not in year) and (pd.to_datetime(row["policy_anniv_dt"]).month not in month):
        #     policy_id.append(row["policy_id"])
        #     year.append(pd.to_datetime(row["policy_anniv_dt"]).year)
        #     month.append(pd.to_datetime(row["policy_anniv_dt"]).month)
        
        #     # if int(date1[4:])<10:
        #     #     date1=date1[:4]+str(0)+date1[4:]
        #     pivot_date.append(date1)
            
        #     x2=pd.to_datetime(row["policy_anniv_dt"])
        #     x1=pd.to_datetime(row["orig_policy_eff_dt"])
        #     survival_month.append(int((x2-x1)/np.timedelta64(1,"M")))
            
        #     policy_anniv_dt.append(row["policy_anniv_dt"])
        #     policy_term_dt.append(row["policy_term_dt"])
        #     orig_policy_eff_dt.append(row["orig_policy_eff_dt"])
            
        #     if pd.to_datetime(row["policy_anniv_dt"])==pd.to_datetime(row["policy_term_dt"]) and row["churn"]==1:
        #         churn.append(1)
        #     else:
        #         churn.append(0)
            
                   
    churn_data=pd.DataFrame({"policy_id":policy_id,"orig_policy_eff_dt":orig_policy_eff_dt,"policy_anniv_dt":policy_anniv_dt,"policy_term_dt":policy_term_dt,
                            "pivot_date":pivot_date,"year":year,"month":month,"survival_month":survival_month,"churn":churn})
    
    churn_data.drop_duplicates(inplace=True)
        # churn_data["month"]=output["month"].apply(lambda x: str(x) if x>=10 else str(0)+str(x))
    churn_data['policy_id']=churn_data['policy_id'].astype(int)
    churn_data['year']=churn_data['year'].apply(int)
    churn_data['month']=churn_data['month'].apply(int)
    
    df=pd.merge(churn_data,policy_df,how="inner", on=["policy_id", "year","month"])
    df.sort_values(by=["policy_id","year","month"],ascending=True,inplace=True)
    
    return df

# data_dir="/app/models/dij22"

# df_buffer_0=data_merge_yearly(churn_labels,policy_premium_df, buffer=0)
# df_buffer_1=data_merge_yearly(churn_labels,policy_premium_df, buffer=1)
# df_buffer_2=data_merge_yearly(churn_labels,policy_premium_df, buffer=2)
# df_buffer_3=data_merge_yearly(churn_labels,policy_premium_df, buffer=3)

# data_dir="/app/models/dij22"
# df_buffer_0.to_pickle(os.path.join(data_dir,"df_buffer_0_pickle"))
# df_buffer_1.to_pickle(os.path.join(data_dir,"df_buffer_1_pickle"))
# df_buffer_2.to_pickle(os.path.join(data_dir,"df_buffer_2_pickle"))
# df_buffer_3.to_pickle(os.path.join(data_dir,"df_buffer_3_pickle"))


data_dir="/app/models/dij22"

# df_buffer_0_hist_3=data_merge_monthly(churn_labels,policy_premium_df,fixed_window=3, buffer=0)
# df_buffer_0_hist_3.to_pickle(os.path.join(data_dir,"df_buffer_0_hist_3_pickle"))

# df_buffer_1_hist_3=data_merge_monthly(churn_labels,policy_premium_df,fixed_window=3, buffer=1)
# df_buffer_1_hist_3.to_pickle(os.path.join(data_dir,"df_buffer_1_hist_3_pickle"))


df_buffer_3_hist_3=data_merge_monthly(churn_labels,policy_premium_df,fixed_window=3, buffer=3)
df_buffer_3_hist_3.to_pickle(os.path.join(data_dir,"df_buffer_3_hist_3_pickle"))

df_buffer_2_hist_3=data_merge_monthly(churn_labels,policy_premium_df,fixed_window=3, buffer=2)
df_buffer_2_hist_3.to_pickle(os.path.join(data_dir,"df_buffer_2_hist_3_pickle"))








