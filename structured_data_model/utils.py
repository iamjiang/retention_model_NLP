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
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
# %matplotlib inline

def model_evaluate(target, predicted):
    
    precision, recall, thresholds = precision_recall_curve(target, predicted)
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    fscore=fscore[~np.isnan(fscore)]
    ix = np.argmax(fscore)
    f1_score=fscore[ix]
    

    auc=roc_auc_score(target, predicted)
    pr_auc=auc_score(recall,precision)

    thrs=thresholds[ix]
    prec=precision[ix]
    reca=recall[ix]

    true_label_mask=[1 if x>=thrs else 0 for i,x in enumerate(predicted)]

    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy=true_prediction/nb_prediction
    
    return {
        "nb_example":len(target),
        "true_prediction":true_prediction,
        "false_prediction":false_prediction,
        "accuracy":accuracy,
        "precision":prec, 
        "recall":reca, 
        "f1_score":f1_score,
        "AUC":auc,
        "pr_auc":pr_auc
    }
    
### Binary Analysis
        
def pcut_func(df,var,nbin=5):
    df[var]=df[var].astype(float)
    df["cut"]=pd.qcut(df[var],nbin,precision=2,duplicates="drop")
    decile=df.groupby(df["cut"])['churn'].mean().reset_index()
    decile["cut"]=decile["cut"].astype(str)
    return decile

def myplot(df,var,*args):

    fig, a = plt.subplots(len(args)//2,2,figsize=(12,2.5*len(args)))
    a=a.ravel()
    for idx,ax in enumerate(a):
      df=args[idx]
      ax.plot(df["cut"],df["churn"],color="r",marker="*",linewidth=2, markersize=12)
      ax.set_title(var[idx])
      ax.tick_params(labelrotation=45)
    fig.tight_layout()

def hist_plot(df,var,r):

    fig, a = plt.subplots(len(var)//2,2,figsize=(12,2*len(var)))
    a=a.ravel()
    for idx,ax in enumerate(a):
      
      ax.hist(df.loc[:,var[idx]], bins=20,range=r)
      ax.set_title(var[idx])
      ax.set_xlabel(var[idx])
      ax.set_ylabel("Frequency")
    fig.tight_layout()
    
    
# https://www.kaggle.com/code/somang1418/tuning-hyperparameters-under-10-minutes-lgbm
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False,metrics="binary_logloss"):
    # prepare data
    train_data = lightgbm.Dataset(data=X, label=y, free_raw_data=False)

    # parameters
    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):
        params = {'application':'binary', 'metric':metrics}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['subsample'] = max(min(subsample, 1), 0)
        params["verbose"]=-1
        
        
        cv_result = lightgbm.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =-1, metrics=[f'{metrics}'])
        if metrics in ["auc"]:
            return max(cv_result[f'{metrics}-mean'])
        elif metrics in ["binary_logloss"]:
            return min(cv_result[f'{metrics}-mean'])
     
    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),
                                            'num_leaves': (24, 80),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.5, 1),
                                            'max_depth': (5, 30),
                                            'max_bin':(20,90),
                                            'min_data_in_leaf': (10, 100),
                                            'min_sum_hessian_in_leaf':(0,100),
                                           'subsample': (0.01, 1.0)}, random_state=200)

    
    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
    
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    model_metr=[]
    for model in range(len( lgbBO.res)):
        model_metr.append(lgbBO.res[model]['target'])
    
    # return best parameters
    opt_params = lgbBO.res[pd.Series(model_metr).idxmax()]['target'],lgbBO.res[pd.Series(model_metr).idxmax()]['params']
    opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
    opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
    opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))
    opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
    opt_params[1]['objective']='binary'
    opt_params[1]['metric']=metrics
    opt_params[1]['is_unbalance']=True
    opt_params[1]['boost_from_average']=False
    opt_params=opt_params[1]
    return opt_params

