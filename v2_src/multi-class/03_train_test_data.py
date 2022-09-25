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
    def __init__(self, df, **kwargs):
        self.df=df
        self.kwargs=kwargs
        
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
        self.df["new_category"]=None
        for k,v in self.kwargs.items():
            
            _, _index=self.subtype_find(keyword=v)
            self.df.loc[_index,["new_category"]]=k
            
        pos_sample=self.df.dropna(subset=["new_category"])
        neg_sample=self.df[self.df["new_category"].isnull()==True].reset_index()
        
        neg_num=pos_sample.shape[0]//len(self.kwargs)
        neg_sample=neg_sample.sample(n=neg_num, random_state=101)
        neg_sample["new_category"]="other-category"
        
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
        
        category_type=sample_data["new_category"].unique().tolist()
        for idx, v in enumerate(category_type):
            data=sample_data[sample_data["new_category"]==v]
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
        
        train_df.to_csv(os.path.join(args.output_dir ,'train_df.csv'))
        val_df.to_csv(os.path.join(args.output_dir ,'val_df.csv'))
        test_df.to_csv(os.path.join(args.output_dir ,'test_df.csv'))
        
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
    
    askunum_text=pd.read_pickle(os.path.join(args.input_dir,"askunum_text"))
    askunum_text['Subtype'] = askunum_text['Subtype'].fillna("").astype(str).str.lower()
    askunum_text["Subtype"]=askunum_text["Subtype"].progress_apply(lambda x: x.encode("latin1").decode("cp1252"))
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("/"," or ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace("&"," and ")
    askunum_text["Subtype"]=askunum_text["Subtype"].str.replace(r"\s{2,}", " ", regex=True)
    
    kwargs={}
    kwargs["billing_issue"]=["bill","billing"]
    kwargs["claim_issue"]=["claim","claims"]
    kwargs["eoi_issue"]=["eoi"]
    kwargs["new_plan_admin"]=["admin","administrator"]

    
    sample_class=Sample_Creation(askunum_text, **kwargs)
    train_df, val_df, test_df=sample_class.data_creation(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    def label_distribution(df,col):
        tempt1=pd.DataFrame(df[col].value_counts(dropna=False)).reset_index().rename(columns={'index':col,col:'count'})
        tempt2=pd.DataFrame(df[col].value_counts(dropna=False,normalize=True)).reset_index().rename(columns={'index':col,col:'percentage'})
        return tempt1.merge(tempt2, on=col, how="inner")

    def style_format(df, col, data_type="Training set"):
        return df.style.format({'count':'{:,}','percentage':'{:.2%}'})\
               .set_caption(f"{data_type} {col} distribution")\
               .set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}])

    label_train=label_distribution(train_df,col="new_category")
    style_format(label_train,col="new_category",  data_type="Training set")
    
    label_train=label_distribution(test_df,col="new_category")
    style_format(label_train,col="new_category",  data_type="Test set")
    
    wrapper = textwrap.TextWrapper(width=150) 
    # Randomly choose some examples.
    for i in range(10):
        random.seed(101+i)

        j = random.choice(train_df.index)
        emails=train_df.loc[j,"TextBody"]
        subtype=train_df.loc[j,"Subtype"]
        category=train_df.loc[j,"new_category"]
        unum_id=train_df.loc[j,"unum_id"]

        print('')
        print("*"*80)
        print(f'*  Full TextBody : unum_id={unum_id}, category={category}, subtype={subtype} *')
        print("*"*80)
        print('')
        # print(j)
        print(wrapper.fill(emails))
        print('')
        print("*"*50)
        
