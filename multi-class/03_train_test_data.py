import argparse
import time
import pandas as pd
import numpy as np
import re
import os
import pickle
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import random
import textwrap

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Sample_Creation:
    def __init__(self, df, *args):
        self.df=df
        self.args=args
        
    def subtype_find(self,keyword):
        subtype=self.df["Subtype"].values.tolist()
        mask=[False for _ in range(len(subtype))]
        for i in range(len(subtype)):
            for k in keyword:
                if k in subtype[i]:
                    mask[i]=True
                    break
        return np.unique(np.array(subtype)[mask]).tolist(),mask
    
    def data_creation(self,val_ratio, test_ratio):
        """"
        Description
        -----------
        create training, validation and test set for positive and negative samples
    
        Parameters:
        ----------
        val_ratio: validation set ratio, default=10% 
        test_ratio: test set ratio, default=10%.  training set ratio=1-val_ratio-test_ratio

        Returns:
        --------
        Traning, validationa and test dataset including positive and negative samples for each category
        
        """        
  
        pos_sample=self.df[self.df["Subtype"].isin(self.args)].reset_index()
        neg_sample=self.df[~self.df["Subtype"].isin(self.args)].reset_index()
        
        neg_num=pos_sample.shape[0]//len(self.args)
        neg_sample=neg_sample.sample(n=neg_num, random_state=101)
        neg_sample["Subtype"]="other-category"
        
        sample_data=pd.concat([pos_sample, neg_sample],axis=0).reset_index()
        
        ## create training, validation and test data based on each category of subtype
        def train_val_test(data,val_ratio,test_ratio):
            data=data.reset_index(drop=True)
            np.random.seed(101)
            _idx=np.arange(len(data))

            np.random.shuffle(_idx)
            test_idx=_idx[:int(len(_idx)*test_ratio)]
            val_idx=_idx[int(len(_idx)*test_ratio) : int(len(_idx)*(val_ratio+test_ratio))]
            train_idx=_idx[int(len(_idx)*(val_ratio+test_ratio)):]
            
            train_data=data.loc[train_idx,:]
            val_data=data.loc[val_idx,:]
            test_data=data.loc[test_idx,:]
            
            return train_data, val_data, test_data
        
        train_df, val_df, test_df=pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        category_type=sample_data["Subtype"].unique().tolist()
        for idx, v in enumerate(category_type):
            data=sample_data[sample_data["Subtype"]==v]
            _train, _val, _test=train_val_test(data,val_ratio,test_ratio)
            train_df=train_df.append(_train)
            val_df=val_df.append(_val)
            test_df=test_df.append(_test)
            
        train_df.drop(['level_0','index'],axis=1,inplace=True)
        val_df.drop(['level_0','index'],axis=1,inplace=True)
        test_df.drop(['level_0','index'],axis=1,inplace=True)
        
        train_df = train_df.sample(frac=1, random_state=101).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=101).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=101).reset_index(drop=True)
        
        train_df.to_csv(os.path.join(args.output_dir ,'train_df_v1.csv'))
        val_df.to_csv(os.path.join(args.output_dir ,'val_df_v1.csv'))
        test_df.to_csv(os.path.join(args.output_dir ,'test_df_v1.csv'))
        
        return train_df, val_df, test_df
    

if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("training, validation and test dataset creation") 
    argparser.add_argument('--output_dir', type=str, default="s3://trident-retention-output/multi-class/")
    argparser.add_argument('--input_dir', type=str, default="s3://trident-retention-output/")
    # argparser.add_argument('--key_subtype', type=str, default=["bill not received","bill hold","bill hide or delete"])
    
    argparser.add_argument('--val_ratio', type=float, default=0.1)
    argparser.add_argument('--test_ratio', type=float, default=0.1)
    
    args,_ = argparser.parse_known_args()
    
    print(args)
    
    askunum_text=pd.read_pickle(os.path.join(args.input_dir,"askunum_text_v1"))
    askunum_text['Subtype'] = askunum_text['Subtype'].fillna("").astype(str).str.lower()
    askunum_text["Subtype"]=askunum_text["Subtype"].progress_apply(lambda x: x.encode("latin1").decode("cp1252"))
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("/"," or ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("&"," and ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace(r"\s{2,}", " ", regex=True)
    
    args_val=['bill hide or delete', "bill not received", "late notice or collections", "missing or skipped payment",'policy level discrepancy', 'premium discrepancy', 
              'broker of record change (bor)','missing information','request to speak to dbs','less than minimum lives', 'policy termination','new plan administrator']

    
    sample_class=Sample_Creation(askunum_text, *args_val)
    train_df, val_df, test_df=sample_class.data_creation(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    def label_distribution(df,col):
        tempt1=pd.DataFrame(df[col].value_counts(dropna=False)).reset_index().rename(columns={'index':col,col:'count'})
        tempt2=pd.DataFrame(df[col].value_counts(dropna=False,normalize=True)).reset_index().rename(columns={'index':col,col:'percentage'})
        return tempt1.merge(tempt2, on=col, how="inner")

    def style_format(df, col, data_type="Training set"):
        return df.style.format({'count':'{:,}','percentage':'{:.2%}'})\
               .set_caption(f"{data_type} {col} distribution")\
               .set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}])

    label_train=label_distribution(train_df,col="Subtype")
    x1=label_train[label_train["Subtype"] != "other-category"]
    x2=label_train[label_train["Subtype"] == "other-category"]
    label_train=pd.concat([x1,x2])
    style_format(label_train,col="Subtype",  data_type="Training set")
    
    label_test=label_distribution(test_df,col="Subtype")
    x1=label_test[label_test["Subtype"] != "other-category"]
    x2=label_test[label_test["Subtype"] == "other-category"]
    label_test=pd.concat([x1,x2])
    style_format(label_test,col="Subtype",  data_type="Test set")
    
    wrapper = textwrap.TextWrapper(width=150) 
    # Randomly choose some examples.
    for i in range(10):
        random.seed(101+i)

        j = random.choice(train_df.index)
        emails=train_df.loc[j,"TextBody"]
        subtype=train_df.loc[j,"Subtype"]

        print('')
        print("*"*80)
        print(f'*  Full TextBody :   subtype={subtype} *')
        print("*"*80)
        print('')
        # print(j)
        print(wrapper.fill(emails))
        print('')
        print("*"*50)

