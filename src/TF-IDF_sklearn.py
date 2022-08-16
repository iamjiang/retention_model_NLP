import argparse
import logging
import os
import time
import numpy as np
from numpy import save,load,savetxt,loadtxt,savez_compressed
import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import shutil
import time
import datetime
import random
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

# from torchtext.data import Field, TabularDataset
# import torchtext.vocab as vocab
# from torchtext.data import Iterator, BucketIterator

from gensim import corpora
from gensim.utils import simple_preprocess

import warnings
warnings.filterwarnings("ignore")

import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

class Batch_Dataset(Dataset):
    def __init__(self,df_tfidf,LABEL,device):
        self.x=df_tfidf.values
        self.y=LABEL
        self.device=device
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]
    
    def make_tfidf_vector(self,batch):
        vec = torch.zeros([len(batch),self.x.shape[1]], dtype=torch.float32, device=self.device)
        target=torch.zeros(len(batch), dtype=torch.long, device=self.device)
        for i,tx in enumerate(batch):
            vec[i]=torch.tensor(tx[0], dtype=torch.float32, device=self.device)
            target[i]=torch.tensor(tx[1], dtype=torch.long, device=self.device)
            
        return vec, target

# %%
# Defining neural network structure
class TFIDF_Classifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size,args):
        # needs to be done everytime in the nn.module derived class
        super(TFIDF_Classifier, self).__init__()
        
        self.args=args
        self.dropout=nn.Dropout(self.args.keep_probab)
        # Define the parameters that are needed for linear model ( Ax + b)
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec): # Defines the computation performed at every call.
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        
        bow_vec=self.dropout(bow_vec)
        
        # clf = LogisticRegression(penalty='l2', C=1.0, random_state=0, max_iter=1000).fit(X, y)
        # clf.predict(X[:2, :])
        # clf.predict_proba(X[:2, :])

        return self.linear(bow_vec)
        # return F.log_softmax(self.linear(bow_vec), dim=1)

def eval_func(data_loader,model,clf,device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    scores=[]
    
    model=model.to(device)
    batch_idx=0
    for bow_vec, target in tqdm(data_loader, position=0, leave=True):
        X = model(bow_vec.float())
        y=target.to(device)
        
        X=X.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
            
        scores.append(clf.score(X, y))
        pred=clf.predict_proba(X)
        
        fin_targets.append(y)
        fin_outputs.append(pred)   

        batch_idx+=1

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), scores

def main(args,train_data, test_data, device):
    train_data.set_format(type="pandas")
    df_train=train_data[:]
    test_data.set_format(type="pandas")
    df_test=test_data[:]
    
    ## undersample netative sample so that the negative/positive=4
    df_train=utils.under_sampling(df_train,'churn', args.random_seed, args.train_negative_positive_ratio)
    df_test=utils.under_sampling(df_test,'churn', args.random_seed, args.test_negative_positive_ratio)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    train_idx=np.arange(df_train.shape[0])
    test_idx=np.arange(df_test.shape[0]) + df_train.shape[0]
    df=pd.concat([df_train,df_test])

    def dummy_fun(doc):
        return doc.split()

    vectorizer = TfidfVectorizer(input="content",analyzer='word',tokenizer=dummy_fun)
    vectorizer.fit(df[args.feature_name])
    df_tfidf = vectorizer.transform(df[args.feature_name])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
    
    LABEL=df["churn"].values.squeeze()
    train_label=LABEL[train_idx]
    test_label=LABEL[test_idx]
    num_classes=np.unique(train_label).shape[0]

    tfidf_train=df_tfidf.iloc[train_idx]
    tfidf_test=df_tfidf.iloc[test_idx]
    
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        
    vocab_size=df_tfidf.shape[1]
    
    model = TFIDF_Classifier(num_classes, vocab_size, args)
    model.to(device)
    
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    no_decay = ["linear.bias"]

    optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    print()
    print("The total # of parameter is {:,}".format(sum([p.nelement() for p in model.parameters() if p.requires_grad]) ) )
    print()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print("{:<70}{:<15,}".format(name,parameter.nelement()))
    print()
    
    train_df=Batch_Dataset(tfidf_train,train_label,device)
    test_df=Batch_Dataset(tfidf_test,test_label,device)

    train_dl = DataLoader(train_df, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_df.make_tfidf_vector)
    test_dl = DataLoader(test_df, batch_size=args.test_batch_size, shuffle=True, collate_fn=test_df.make_tfidf_vector)


    print('{:<20} {:<10,}'.format("Train loader",len(train_dl)))
    print('{:<20} {:<10,}'.format("validation loader",len(test_dl)))
    

    # training loop
    # best_metric = float('inf')
    best_metric = 0
    scores=[]

    for epoch in tqdm(range(0,args.n_epochs)):

        model.train()

        LOGIT_train=[]
        LABEL_train=[]
        #====================================#
        #            Traning                 #
        #====================================#
        print("")
        print("========= Epoch {:} /{:}".format(epoch+1,args.n_epochs))
        print("Training...")
        t0 = time.time()

        for step, (bow_vec, target) in enumerate(train_dl):

            X = model(bow_vec.float())
            y=target.to(device)
            
            X=X.cpu().detach().numpy()
            y=y.cpu().detach().numpy()

            if loss_weight is None:
                clf = LogisticRegression(penalty='l2', C=args.C, random_state=0, max_iter=5000).fit(X, y)
            else:
                clf = LogisticRegression(penalty='l2', C=args.C, random_state=0, max_iter=5000,class_weight={0:loss_weight[0],1:loss_weight[1]}).fit(X, y)

            scores.append(clf.score(X, y))

            t1 = time.time()
            if step%(len(train_dl)//10)==0 and not step==0:
                elapsed=utils.format_time(t1-t0)
                print("Batch {:} of {:} | Score {:.3f}  | Elapsed: {:}".\
                      format(step,len(train_dl),np.mean(scores[-10:]),elapsed))    

        model.eval()
        print()
        print("")
        print("Running Validation on training set")
        print("")
        t1=time.time()
        train_pred,train_target,train_score=eval_func(train_dl,model,clf,device)

        avg_train_score=np.mean(train_score)
        train_output=utils.model_evaluate(train_target.reshape(-1),train_pred)

        t2=time.time()

        print("avg_score: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} \
        ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_train_score, train_output["true_prediction"], train_output["false_prediction"], train_output["accuracy"], \
                                                                train_output["precision"], train_output["recall"], train_output["f1_score"], \
                                                                train_output["AUC"], train_output["pr_auc"], utils.format_time(t2-t1)))

        gain_1=train_output["GAIN"]["1%"]
        gain_5=train_output["GAIN"]["5%"]
        gain_10=train_output["GAIN"]["10%"]
        with open(os.path.join(os.getcwd(),"metrics_training.txt"),'a') as f:
            f.write(f'{args.model_output_name},{epoch},{avg_train_score},{train_output["true_prediction"]},{train_output["false_prediction"]},{train_output["accuracy"]},{train_output["precision"]},{train_output["recall"]},\
            {train_output["f1_score"]},{gain_1},{gain_5},{gain_10},{train_output["AUC"]},{train_output["pr_auc"]}\n')
                  
        #====================================#
        #            Test-set          #
        #====================================#

        model.eval()
        print()
        print("")
        print("Running Validation on test set")
        print("")
        test_pred,test_target,test_score=eval_func(test_dl,model,clf,device)

        avg_test_score=np.mean(test_score)
        test_output=utils.model_evaluate(test_target.reshape(-1),test_pred)

        t3=time.time()

        print("avg_score: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} \
        ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_test_score, test_output["true_prediction"], test_output["false_prediction"], test_output["accuracy"], \
                                                                test_output["precision"], test_output["recall"], test_output["f1_score"], \
                                                                test_output["AUC"], test_output["pr_auc"], utils.format_time(t3-t2)))

        gain_1=test_output["GAIN"]["1%"]
        gain_5=test_output["GAIN"]["5%"]
        gain_10=test_output["GAIN"]["10%"]
        with open(os.path.join(os.getcwd(),"metrics_test.txt"),'a') as f:
            f.write(f'{args.model_output_name},{epoch},{avg_test_score},{test_output["true_prediction"]},{test_output["false_prediction"]},{test_output["accuracy"]},{test_output["precision"]},{test_output["recall"]},\
            {test_output["f1_score"]},{gain_1},{gain_5},{gain_10},{test_output["AUC"]},{test_output["pr_auc"]}\n')    
    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        selected_metric=avg_test_score
        if selected_metric>best_metric:
            best_metric=selected_metric
            torch.save(model,os.path.join(args.output_dir,"tfidf.pt"))
            print("")
            logger.info(f'Performance improve after epoch: {epoch+1} ... ')
            print("")   
            
#             model = torch.load(os.path.join(args.output_dir,"tfidf.pt"))
#             model.to(device)
#             model.eval()
                
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="TF-IDF Feature")
    parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=101)
    
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=4,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
#     parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--keep_probab', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--C', type=float, default=0.2, help="regularization strength")
    
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--loss_weight',  action='store_true')
    parser.add_argument("--feature_name", default="Full_TextBody", type=str)
    parser.add_argument("--model_output_name", default="TF-IDF", type=str)
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(),"TF-IDF"), type=str, help="output folder name")
    
    args= parser.parse_args()
    
    args.model_output_name=f'{args.model_output_name}_{args.feature_name}'
    args.output_dir=f'{args.output_dir}_{args.feature_name}'
    
    print()
    print(args)
    print()
    
    seed_everything(args.random_seed)
    
    data_dir=os.path.join(os.getcwd(),"dataset","email_all")
    email_all=load_from_disk(data_dir)
    email_all=email_all.filter(lambda x: x[args.feature_name]!=None)
    
    train_data=email_all['train']
    test_data=email_all['test']
#     train_data=email_all['train'].shuffle(seed=101).select(range(1200))
#     test_data=email_all['test'].shuffle(seed=101).select(range(500))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)
    # print(f"The number of GPUs is {torch.cuda.device_count()}")
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print()
        print('{:<30}{:<10}'.format("The # of availabe GPU(s): ",torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            print('{:<30}{:<10}'.format("GPU Name: ",torch.cuda.get_device_name(i)))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
        
    main(args,train_data, test_data, device)
    