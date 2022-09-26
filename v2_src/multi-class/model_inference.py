import os
import numpy as np
import pandas as pd
# from tqdm.auto import tqdm
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import random
import argparse
import logging

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, label_binarize

import textwrap

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

import transformers
print("Transformers version is {}".format(transformers.__version__))

import torch
from torch.utils.data import DataLoader, RandomSampler

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
    get_scheduler
)

import utils

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--shuffle_train",  type=bool,default=True,help="shuffle data or not")
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path",type=str,default="/home/ec2-user/SageMaker/retention_model_NLP/v2_src/multi-class/roberta_large_repo")
    parser.add_argument("--feature_name", default="TextBody", type=str)
    parser.add_argument("--is_train_inference", action="store_true", help="undersampling or not")


    args= parser.parse_args()
    
    print(args)
    seed_everything(args.seed)
    
    input_dir="s3://trident-retention-output/"
    output_dir="s3://trident-retention-output/multi-class/"

    askunum_text=pd.read_pickle(os.path.join(input_dir,"askunum_text"))
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

    sample_class=utils.Sample_Creation(askunum_text, **kwargs)
    train_df,val_df,test_df=sample_class.data_creation(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    
    seed_everything(args.seed)

    def cate_2_int_label(df,col):
        uniq_label=df[col].unique()
        uniq_label.sort()
        label_map={v:idx for idx,v in enumerate(uniq_label)}
        df[col]=list(map(label_map.get, df[col]))
        df = df.rename(columns={col: 'label'})
        return df, label_map

    train_df, train_label_map=cate_2_int_label(train_df,col="new_category")
    val_df, val_label_map=cate_2_int_label(val_df,col="new_category")
    test_df,  test_label_map=cate_2_int_label(test_df,col="new_category")

    # train_df=train_df.sample(n=1000)
    # val_df=val_df.sample(n=1000)
    # test_df=test_df.sample(n=1000)

    hf_train=Dataset.from_pandas(train_df)
    hf_val=Dataset.from_pandas(val_df)
    hf_test=Dataset.from_pandas(test_df)
    # hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})
    hf_data=concatenate_datasets([hf_train,  hf_val],split="train")
    hf_data=DatasetDict({"train":hf_data, "test":hf_test})

    hf_data=hf_data.filter(lambda x: x[args.feature_name]!=None)

    train_label=train_df['label'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]

    tokenizer=AutoTokenizer.from_pretrained(args.model_path)
    model=AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels = num_classes)

    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters : {sum([p.nelement() for p in model.parameters()]):,}")
    print()

    hf_data=hf_data.map(lambda x: tokenizer(x[args.feature_name]),batched=True)

    max_seq_length=tokenizer.model_max_length
    def truncation_text(example):
        truncated_input_ids=tokenizer(example[args.feature_name],truncation=True,padding=False,return_tensors="pt",add_special_tokens=False)['input_ids']

        if args.truncation_strategy=="tail":
            truncated_input_ids=truncated_input_ids[:,-(max_seq_length - 2):].squeeze()
        elif args.truncation_strategy=="head":
            truncated_input_ids=truncated_input_ids[:,0:(max_seq_length - 2)].squeeze()
        elif args.truncation_strategy=="mixed":
            truncated_input_ids=truncated_input_ids[:(max_seq_length - 2) // 2] + truncated_input_ids[-((max_seq_length - 2) // 2):]
            truncated_input_ids=truncated_input_ids.squeeze()
        else:
            raise NotImplemented("Unknown truncation. Supported truncation: tail, head, mixed truncation")

        return {"truncated_text":tokenizer.decode(truncated_input_ids)}

    hf_data=hf_data.map(truncation_text)
    columns=hf_data['train'].column_names
    columns_to_keep=['truncated_text','label']
    columns_to_remove=set(columns)-set(columns_to_keep)
    hf_data=hf_data.remove_columns(columns_to_remove)
    hf_data=hf_data.rename_column("truncated_text", args.feature_name)

    train_data=hf_data['train'].shuffle(seed=101).select(range(len(hf_data["train"])))
    # val_data=hf_data['val'].shuffle(seed=101).select(range(len(hf_data["val"])))
    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))

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

    train_data.set_format(type="pandas")
    df_train=train_data[:]
    test_data.set_format(type="pandas")
    df_test=test_data[:]

    train_data=Dataset.from_pandas(df_train)
    test_data=Dataset.from_pandas(df_test)


    train_module=utils.Loader_Creation(train_data, tokenizer,args.feature_name)


    test_module=utils.Loader_Creation(test_data, tokenizer,args.feature_name)

    train_data.set_format(type="pandas")
    df_train=train_data[:]
    train_data.reset_format()

    train_dataloader=DataLoader(train_module,
                                shuffle=True,
                                batch_size=args.batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=test_module.collate_fn
                               )

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    #     print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))


    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        
    
    if args.is_train_inference:
        y_pred, y_target, losses_tmp=utils.eval_func(train_dataloader,model,device,num_classes=num_classes,loss_weight=loss_weight)
        label_map={v:k for k,v in train_label_map.items()}
    else:
        y_pred, y_target, losses_tmp=utils.eval_func(test_dataloader,model,device,num_classes=num_classes,loss_weight=loss_weight)
        label_map={v:k for k,v in test_label_map.items()}
    
    metrics_dict, roc_auc, pr_auc = utils.model_evaluate(y_target,y_pred)

    print("{:<20}{:<10.2%}".format("accuracy", metrics_dict['acc']))
    print()
    print("{:<20}{:<10,.2%}{:<16}{:<10,.2%}{:<18}{:<10,.2%}{:<17}{:<10,.2%}{:<16}{:<10,.2%}"\
          .format("precision(macro):",metrics_dict['prec_macro'],"recall(macro):",metrics_dict['recall_macro'],\
                  "f1-score(macro):",metrics_dict['fscore_macro'],"ROC-AUC(macro):",metrics_dict['auc_macro_ovo'],\
                 "PR-AUC(macro):",metrics_dict['pr_auc_macro']))

    print("{:<20}{:<10,.2%}{:<16}{:<10,.2%}{:<18}{:<10,.2%}{:<17}{:<10,.2%}{:<16}{:<10,.2%}"\
          .format("precision(micro):",metrics_dict['prec_micro'],"recall(micro):",metrics_dict['recall_micro'],\
                  "f1-score(micro):",metrics_dict['fscore_micro'],"ROC-AUC(micro):",metrics_dict['auc_micro'],\
                 "PR-AUC(micro):",metrics_dict['pr_auc_micro']))

    print("{:<20}{:<10,.2%}{:<16}{:<10,.2%}{:<18}{:<10,.2%}{:<17}{:<10,.2%}{:<16}{:<10,.2%}"\
          .format("precision(weight):",metrics_dict['prec_weighted'],"recall(weight):",metrics_dict['recall_weighted'],\
                  "f1-score(weight):",metrics_dict['fscore_weighted'],"ROC-AUC(weight):",metrics_dict['auc_weighted_ovo'],\
                 "PR-AUC(weight):",metrics_dict['pr_auc_weighted']))

    print()
    print(label_map)
    
    n_classes=len(label_map)
    
    report=metrics.classification_report(y_target.squeeze(), y_pred.argmax(axis=1), output_dict=True)

    table = pd.DataFrame(report).transpose().iloc[:n_classes,:]
    table["count"]=table["support"].astype(int)
    table["roc_auc"]=[roc_auc[i] for i in range(n_classes)]
    table["pr_auc"]=[pr_auc[i] for i in range(n_classes)]
    table["subtype_type"]=[label_map[i] for i in range(n_classes)]
    table=table[['subtype_type','count','precision','recall','f1-score','roc_auc','pr_auc']]

    total=table['count'].sum()

    table.loc[len(table.index)]=["MACRO",total,metrics_dict['prec_macro'],metrics_dict['recall_macro'],metrics_dict['fscore_macro'],\
                            metrics_dict['auc_macro_ovo'],metrics_dict['pr_auc_macro']]

    table.loc[len(table.index)]=["MICRO",total,metrics_dict['prec_micro'],metrics_dict['recall_micro'],metrics_dict['fscore_micro'],\
                                metrics_dict['auc_micro'],metrics_dict['pr_auc_micro']]

    table.loc[len(table.index)]=["WEIGHT",total,metrics_dict['prec_weighted'],metrics_dict['recall_weighted'],metrics_dict['fscore_weighted'],\
                            metrics_dict['auc_weighted_ovo'],metrics_dict['pr_auc_weighted']]

    table.style.format({"count":"{:,}","f1-score":"{:.2%}","precision":"{:.2%}","recall":"{:.2%}","roc_auc":"{:.2%}","pr_auc":"{:.2%}"})
    
    