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

class Sample_Creation:
    def __init__(self, df, subtype, keyword):
        self.df=df
        self.subtype=subtype
        self.keyword=keyword
        
    def subtype_find(self):
        mask=[False for _ in range(len(self.subtype))]
        for i in range(len(self.subtype)):
            for k in self.keyword:
                if k in self.subtype[i]:
                    mask[i]=True
                    break
        return np.array(self.subtype)[mask].tolist()
    
    def data_creation(self,val_ratio, test_ratio, train_pos_neg_ratio, val_pos_neg_ratio, test_pos_neg_ratio):
        """"
        Description
        -----------
        create training, validation and test set for positive and negative samples
    
        Parameters:
        ----------
        val_ratio: validation set ratio, default=10% 
        test_ratio: test set ratio, default=10%.  training set ratio=1-val_ratio-test_ratio
        train_pos_neg_ratio : The proportion of positive sample vs negative sampel in training set
        val_pos_neg_ratio :   The proportion of positive sample vs negative sampel in validation set
        test_pos_neg_ratio :  The proportion of positive sample vs negative sampel in test set

        Returns:
        --------
        Traning, validationa and test dataset including positive and negative samples
        
        """
        _subtype=self.subtype_find()
        pos_sample=self.df[self.df['Subtype'].isin(_subtype)].reset_index()
        neg_sample=self.df[~self.df['Subtype'].isin(_subtype)].reset_index()
        
        def train_val_test(data,val_ratio,test_ratio):
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
        
        train_positive, val_positive, test_positive=train_val_test(pos_sample,val_ratio,test_ratio)
        train_negative, val_negative, test_negative=train_val_test(neg_sample,val_ratio,test_ratio)
        
        train_neg_num=len(train_positive)* train_pos_neg_ratio
        val_neg_num=len(val_positive)* val_pos_neg_ratio
        test_neg_num=len(test_positive)* test_pos_neg_ratio
        
        train_negative=train_negative.sample(n=train_neg_num, random_state=101)
        val_negative=val_negative.sample(n=val_neg_num, random_state=101)
        test_negative=test_negative.sample(n=test_neg_num, random_state=101)
        
        train_positive["label"]=1
        val_positive["label"]=1
        test_positive["label"]=1
        
        train_negative["label"]=0
        val_negative["label"]=0
        test_negative["label"]=0
        
        train_df=pd.concat([train_positive, train_negative],axis=0).reset_index()
        val_df=pd.concat([val_positive, val_negative],axis=0).reset_index()
        test_df=pd.concat([test_positive, test_negative],axis=0).reset_index()
        
        train_df.drop(['level_0','index'],axis=1,inplace=True)
        val_df.drop(['level_0','index'],axis=1,inplace=True)
        test_df.drop(['level_0','index'],axis=1,inplace=True)
        
        train_df.to_csv(os.path.join(args.output_dir ,'train_df.csv'))
        val_df.to_csv(os.path.join(args.output_dir ,'val_df.csv'))
        test_df.to_csv(os.path.join(args.output_dir ,'test_df.csv'))
        return train_df, val_df, test_df
        

if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("training, validation and test dataset creation") 
    argparser.add_argument('--input_dir', type=str, default="s3://trident-retention-data/askunum/")
    argparser.add_argument('--output_dir', type=str, default="s3://trident-retention-output/")
    argparser.add_argument('--key_subtype', type=str, default=["bill not received","bill hold","bill hide or delete"])
    
    argparser.add_argument('--val_ratio', type=float, default=0.1)
    argparser.add_argument('--test_ratio', type=float, default=0.1)
    argparser.add_argument('--train_pos_neg_ratio', type=int, default=3)
    argparser.add_argument('--val_pos_neg_ratio', type=int, default=3)
    argparser.add_argument('--test_pos_neg_ratio', type=int, default=9)
    
    args,_ = argparser.parse_known_args()
    
    print(args)
    
    askunum_text=pd.read_pickle(os.path.join(args.output_dir,"askunum_text"))
    askunum_text['Subtype'] = askunum_text['Subtype'].fillna("").astype(str).str.lower()
    askunum_text["Subtype"]=askunum_text["Subtype"].progress_apply(lambda x: x.encode("latin1").decode("cp1252"))
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("/"," or ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("&"," and ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace(r"\s{2,}", " ", regex=True)
    
    df=askunum_text[~askunum_text["Subtype"].isin(["attempted self-service - billing support"])]
    subtype=list(df["Subtype"].unique())
    # keyword=["bill not received","bill hold","bill hide or delete"]
    sample_class=Sample_Creation(df, subtype, args.key_subtype)
    train_df, val_df, test_df=sample_class.data_creation(val_ratio=args.val_ratio, test_ratio=args.test_ratio, \
                                                         train_pos_neg_ratio=args.train_pos_neg_ratio, \
                                                         val_pos_neg_ratio=args.val_pos_neg_ratio, \
                                                         test_pos_neg_ratio=args.test_pos_neg_ratio)
    
    
    wrapper = textwrap.TextWrapper(width=150) 
    # Randomly choose some examples.
    for i in range(5):
        random.seed(101+i)

        j = random.choice(train_df.index)
        emails=train_df.loc[j,"TextBody"]
        subtype=train_df.loc[j,"Subtype"]
        unum_id=train_df.loc[j,"unum_id"]

        print('')
        print("*"*80)
        print(f'*  Full TextBody : unum_id={unum_id}, subtype={subtype} *')
        print("*"*80)
        print('')
        # print(j)
        print(wrapper.fill(emails))
        print('')
        print("*"*50)
        
