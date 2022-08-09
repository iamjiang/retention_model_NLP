import argparse
import os
import logging
import math
import time
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import shutil
import time
import datetime
import random
import math
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

# os.system("python -m spacy download en_core_web_md")
import spacy
nlp = spacy.load("en_core_web_md")

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

from transformers import AdamW, get_scheduler

from torchtext import data
import torchtext.vocab as vocab

import utils

import warnings
warnings.filterwarnings("ignore")

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
    
class CNN_Model(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, pretrained_emb, mode):
        super(CNN_Model, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        pretrained_emb : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.mode=mode

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
#         self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        if self.mode=="rand":
            rand_embed_init=torch.Tensor(vocab_size, embedding_length).uniform_(-0.25,0.25)
            self.word_embeddings.weight.data.copy_(rand_embed_init)
            self.word_embeddings.weight.requires_grad = True
        elif self.mode=="static":
            self.word_embeddings.weight.data.copy_(pretrained_emb)
            self.word_embeddings.weight.requires_grad = False
        elif self.mode=="non-static":
            self.word_embeddings.weight.data.copy_(pretrained_emb)
            self.word_embeddings.weight.requires_grad = True
        elif self.mode=="multi-channel":
            self.static_embed=nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            self.non_static_embed=nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
        else:
            print("Unsupport Mode")
            exit()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out
    
    def forward(self, input_sentences,device):

        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """
        if self.mode in ["rand","static","non-static"]: 
            input = self.word_embeddings(input_sentences)
            # input.size() = (batch_size, num_seq, embedding_length)
            input = input.unsqueeze(1).to(device)
            # input.size() = (batch_size, 1, num_seq, embedding_length)
        elif self.mode=="multi-channel":
            static_input=self.static_embed(input_sentences)
            non_static_input=self.non_static_embed(input_sentences)
            input=torch.stack([static_input,non_static_input],dim=1).to(device)
            # input.size() = (batch_size, input_channel=2, num_seq, embedding_length)
        else:
            print("Unsupport Mode")
            exit()

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits
    
    
def load_dataset(df_train, df_test, args, device,custom_embeddings=None):
    df_train=df_train.loc[:,[args.feature_name,"churn"]]
    df_test=df_test.loc[:,[args.feature_name,"churn"]]
    df_train.to_csv("df_train.csv", index=False)
    df_test.to_csv("df_test.csv", index=False)

#     tokenize=lambda x : x.split()
#     TEXT=Field(sequential=True, tokenize='basic_english', lower=True)
    nlp = spacy.load("en_core_web_md")
    TEXT=data.Field(sequential=True, tokenize='spacy', tokenizer_language = 'en_core_web_md',include_lengths=False)
    LABEL=data.Field(sequential=False,use_vocab=False)

    tv_datafields=[(args.feature_name,TEXT),('churn',LABEL)]

    trn, vld=data.TabularDataset.splits(path="./",\
                                   train="df_train.csv",\
                                   validation="df_test.csv",\
                                   format="csv",\
                                   skip_header=True,\
                                   fields=tv_datafields)

    os.system("rm *.csv")

    if args.is_pretrain_embedding:
        TEXT.build_vocab(trn, vld,vectors=custom_embeddings)
    else:
        TEXT.build_vocab(trn, vld)
    
#     LABEL.build_vocab(trn)
    
    word_embeddings=TEXT.vocab.vectors
    
    print()
    print ("{:<40}{:<15,}".format("Length of Text Vocabulary: ", len(TEXT.vocab)))
    print ("{:<40}{:}".format("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size()))
    print()
    
    #### Creating the iterator
    train_iter,_=data.BucketIterator.splits((trn,vld),
                                             batch_sizes=(args.train_batch_size, args.test_batch_size),
                                             device=device,
                                             sort_key=lambda x : len(x[args.feature_name]),
                                             sort_within_batch=False,
                                             repeat=False)
    
    valid_iter,_=data.BucketIterator.splits((vld,trn),
                                            batch_sizes=(args.test_batch_size,args.train_batch_size),
                                            device=device,
                                            sort_key=lambda x : len(x[args.feature_name]),
                                            sort_within_batch=False,
                                            repeat=False)
    
    vocab_size = len(TEXT.vocab)
    
    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter 

def eval_func(data_loader,model,device,num_classes=2,loss_weight=None):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    
    model=model.to(device)
#     for batch_idx, batch in enumerate(data_loader):
    batch_idx=0
    for step, batch in enumerate(data_loader):
        text=getattr(batch,args.feature_name)
        input_sentences=text.transpose(0,1)
        target=getattr(batch,'churn')
        with torch.no_grad():
            logits = model(input_sentences,device)

        if loss_weight is None:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), target.to(device))
        else:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), target.to(device), weight=loss_weight.float().to(device))
            
        losses.append(loss.item())
        
        fin_targets.append(target.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())   

        batch_idx+=1

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=torch.sum(y.squeeze()==i).item()
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

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
    
    custom_embeddings = vocab.Vectors(name="glove.6B.300d.txt", cache=os.getcwd())
    
    TEXT, vocab_size, word_embeddings, train_iter, test_iter = load_dataset(df_train, df_test, args, device, custom_embeddings=custom_embeddings)
    
    train_label=[]
    for batch in tqdm(train_iter, position=0, leave=True):
        text=getattr(batch,args.feature_name)
        input_sentences=text.transpose(0,1)
        target=getattr(batch,'churn')
        train_label.append(target)
    train_label=torch.cat(train_label)
    num_classes=torch.unique(train_label).shape[0]

    if args.loss_weight:
        train_classes_num, train_classes_weight = get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        
    test_label=[]
    for batch in tqdm(test_iter, position=0, leave=True):
        text=getattr(batch,args.feature_name)
        input_sentences=text.transpose(0,1)
        target=getattr(batch,'churn')
        test_label.append(target)
    test_label=torch.cat(test_label)
    print()
    print('{:<15} {:<10,}'.format("Train loader",len(train_iter)))
    print('{:<15} {:<10,}'.format("Test loader",len(test_iter)))
    print()  
    
    vocab_size= len(TEXT.vocab)
    pretrained_emb=TEXT.vocab.vectors

    model=CNN_Model(args.train_batch_size, num_classes, args.in_channels, args.out_channels, args.kernel_heights, args.stride, args.padding, args.keep_probab, \
                    vocab_size, args.embedding_length, pretrained_emb, args.mode)
    model.to(device)

    no_decay = ["conv1.bias","conv2.bias","conv3.bias","label.bias"]

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
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = int((len(train_iter) // args.train_batch_size)//args.gradient_accumulation_steps*float(args.n_epochs))
    warmup_steps=int((len(train_iter) // args.train_batch_size)//args.gradient_accumulation_steps*args.warmup_ratio)
    
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=t_total)
    
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    print()
    print("The total # of parameter is {:,}".format(sum([p.nelement() for p in model.parameters() if p.requires_grad]) ) )
    print()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print("{:<70}{:<15,}".format(name,parameter.nelement()))
    print()
    
    best_metric = float('inf')
    # best_metric = 0
    losses=[]

    for epoch in tqdm(range(0,args.n_epochs),position=0 ,leave=True):

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

        for step, batch in enumerate(train_iter):
    
            text=getattr(batch,args.feature_name)
            input_sentences=text.transpose(0,1)
            target=getattr(batch,'churn')
            logits = model(input_sentences,device)

            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes), target.to(device))
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), target.to(device),weight=loss_weight.float())

            loss.backward()
            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_iter)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())

            t1 = time.time()
            if step%(len(train_iter)//10)==0 and not step==0:
                elapsed=utils.format_time(t1-t0)
                print("Batch {:} of {:} | Loss {:.3f}  | Elapsed: {:}".\
                      format(step,len(train_iter),np.mean(losses[-10:]),elapsed))    

        model.eval()
        print()
        print("")
        print("Running Validation on training set")
        print("")
        t1=time.time()
        train_pred,train_target,train_losses=eval_func(train_iter,
                                                       model,
                                                       device,
                                                       num_classes=num_classes, 
                                                       loss_weight=loss_weight)

        avg_train_loss=np.mean(train_losses)
        train_output=utils.model_evaluate(train_target.reshape(-1),train_pred)

        t2=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} \
        ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_train_loss, train_output["true_prediction"], train_output["false_prediction"], train_output["accuracy"], \
                                                                train_output["precision"], train_output["recall"], train_output["f1_score"], \
                                                                train_output["AUC"], train_output["pr_auc"], utils.format_time(t2-t1)))

        gain_1=train_output["GAIN"]["1%"]
        gain_5=train_output["GAIN"]["5%"]
        gain_10=train_output["GAIN"]["10%"]
        with open(os.path.join(os.getcwd(),"metrics_training.txt"),'a') as f:
            f.write(f'{args.model_output_name},{epoch},{avg_train_loss},{train_output["true_prediction"]},{train_output["false_prediction"]},{train_output["accuracy"]},{train_output["precision"]},{train_output["recall"]},\
            {train_output["f1_score"]},{gain_1},{gain_5},{gain_10},{train_output["AUC"]},{train_output["pr_auc"]}\n')
                  
        #====================================#
        #            Test-set          #
        #====================================#

        model.eval()
        print()
        print("")
        print("Running Validation on test set")
        print("")

        test_pred,test_target,test_losses=eval_func(test_iter,
                                                       model,
                                                       device,
                                                       num_classes=num_classes, 
                                                       loss_weight=loss_weight)

        avg_test_loss=np.mean(test_losses)
        test_output=utils.model_evaluate(test_target.reshape(-1),test_pred)

        t3=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} \
        ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_test_loss, test_output["true_prediction"], test_output["false_prediction"], test_output["accuracy"], \
                                                                test_output["precision"], test_output["recall"], test_output["f1_score"], \
                                                                test_output["AUC"], test_output["pr_auc"], utils.format_time(t3-t2)))

        gain_1=test_output["GAIN"]["1%"]
        gain_5=test_output["GAIN"]["5%"]
        gain_10=test_output["GAIN"]["10%"]
        with open(os.path.join(os.getcwd(),"metrics_test.txt"),'a') as f:
            f.write(f'{args.model_output_name},{epoch},{avg_test_loss},{test_output["true_prediction"]},{test_output["false_prediction"]},{test_output["accuracy"]},{test_output["precision"]},{test_output["recall"]},\
            {test_output["f1_score"]},{gain_1},{gain_5},{gain_10},{test_output["AUC"]},{test_output["pr_auc"]}\n')    
    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        selected_metric=avg_test_loss
        if selected_metric<best_metric:
            best_metric=selected_metric
            torch.save(model,os.path.join(args.output_dir,"tfidf.pt"))
            print("")
            logger.info(f'Performance improve after epoch: {epoch+1} ... ')
            print("")   
            
#             model = torch.load(os.path.join(args.output_dir,"tfidf.pt"))
#             model.to(device)
#             model.eval()
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="CNN Model")
    parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=101)
    parser.add_argument('--is_pretrain_embedding',  action='store_true')
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=3,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=3,help="Undersampling negative vs position ratio in test set")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
#     parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=100)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--mode', type=str, default="non-static",help="mode to update word to vector embedding")
    parser.add_argument('--kernel_heights', type=int, default=[3,4,5], nargs='+', help='kernel size')
    parser.add_argument('--keep_probab', type=float, default=0.2)
    parser.add_argument('--embedding_length', type=int, default=300)
    
    parser.add_argument('--loss_weight',  action='store_true')
    parser.add_argument("--feature_name", default="Full_TextBody", type=str)
    parser.add_argument("--model_output_name", default="CNN", type=str)
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(),"CNN"), type=str, help="output folder name")
    
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
        
        
#     os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
#     os.system("unzip glove*.zip")
    
#     os.system("rm glove.6B.50d.txt")
#     os.system("rm glove.6B.100d.txt")
#     os.system("rm glove.6B.200d.txt")
#     os.system("rm *.zip")
    
    
    main(args,train_data, test_data, device)
    
    