import os
import time
import datetime
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import logging

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

import transformers
print("Transformers version is {}".format(transformers.__version__))

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

from accelerate import Accelerator

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
    
    
def main(args,train_data, test_data, device):
    train_data.set_format(type="pandas")
    df_train=train_data[:]
    test_data.set_format(type="pandas")
    df_test=test_data[:]
    
    ## undersample netative sample so that the negative/positive=4
    if args.undersampling:
        df_train=utils.under_sampling(df_train,'label', args.seed, args.train_negative_positive_ratio)
        df_test=utils.under_sampling(df_test,'label', args.seed, args.test_negative_positive_ratio)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
    
    train_data=Dataset.from_pandas(df_train)
    test_data=Dataset.from_pandas(df_test)
    
    tokenizer=AutoTokenizer.from_pretrained(args.model_checkpoint)
    model=AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint)

    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters : {sum([p.nelement() for p in model.parameters()]):,}")
    print()
    
    train_module=utils.Loader_Creation(train_data, tokenizer,args.feature_name)
    

    test_module=utils.Loader_Creation(test_data, tokenizer,args.feature_name)

    train_data.set_format(type="pandas")
    df_train=train_data[:]
    train_data.reset_format()

#     train_indices, val_indices=utils.mask_creation(df_train, 'label', args.seed, args.validation_split)

    

#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader=DataLoader(train_module,
                                shuffle=True,
                                batch_size=args.batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )
    
#     train_dataloader=DataLoader(train_module,
#                                 sampler=train_sampler,
#                                 batch_size=args.batch_size,
#                                 collate_fn=train_module.collate_fn,
#                                 drop_last=True   # longformer model bug
#                                )

#     valid_dataloader=DataLoader(train_module,
#                                 sampler=valid_sampler,
#                                 batch_size=args.batch_size,
#                                 collate_fn=train_module.collate_fn
#                                )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=test_module.collate_fn
                               )

    # %pdb
    # next(iter(train_dataloader))

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
#     print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    
    train_label=df_train['label'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        

    t_total = int((len(train_dataloader) // args.batch_size)//args.gradient_accumulation_steps*float(args.num_epochs))

    warmup_steps=int((len(train_dataloader) // args.batch_size)//args.gradient_accumulation_steps*args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # optimizer=AdamW(model.parameters(),lr=args.lr)
    #     lr_scheduler =get_linear_schedule_with_warmup(optimizer, 
    #                                                   num_warmup_steps=warmup_steps, 
    #                                                   num_training_steps=t_total
    #                                                  )

    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=t_total)
    
    accelerator = Accelerator(fp16=args.fp16)
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    if accelerator.is_main_process:
        accelerator.print("")
        logger.info(f'Accelerator Config: {acc_state}')
        accelerator.print("")
    
#     model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
#         model, optimizer, train_dataloader, valid_dataloader, test_dataloader
#     )
    
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )
    
    best_metric = float('inf')
    # best_metric = 0
    
    iter_tput = []
    
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
        
        accelerator.print(f"\n===========EPOCH {epoch+1}/{args.num_epochs}===============\n")
        model.train()
                
        losses=[]
        for step,batch in enumerate(train_dataloader):
            t0=time.time()
            batch={k:v.type(torch.LongTensor).to(accelerator.device) for k,v in batch.items()}
            outputs=model(**batch)
#             loss=outputs.loss
            # print(outputs)
            logits=outputs.loss['logits']
            
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(accelerator.device),batch["labels"])
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(accelerator.device),batch["labels"], \
                                       weight=loss_weight.float().to(accelerator.device)) 
            
            accelerator.backward(loss)
            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
                optimizer.step()
                if args.use_schedule:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
            losses.append(loss.item())
            
            iter_tput.append(batch["input_ids"].shape[0] / (time.time() - t0))
            
            if step%(len(train_dataloader)//10)==0 and not step==0 :
                accelerator.print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                                  .format(epoch, step, np.mean(losses[-10:]), np.mean(iter_tput[3:]), 
                                          torch.cuda.max_memory_allocated() / 1000000))

#         epoch_loss=np.mean(losses)
#         accelerator.print(f"\n** avg_loss : {epoch_loss:.2f}, time :~ {(time.time()-t0)//60} min ({time.time()-t0 :.2f} sec)***\n")

        t1=time.time()
        train_pred,train_target,train_losses=utils.eval_func(train_dataloader,
                                                             model, 
                                                             accelerator.device,
                                                             num_classes=num_classes, 
                                                             loss_weight=loss_weight)

        avg_train_loss=np.mean(train_losses)

        train_output=utils.model_evaluate(train_target.reshape(-1),train_pred)


        t2=time.time()
        accelerator.print("")
        accelerator.print("==> Running Validation on training set \n")
        accelerator.print("")
        accelerator.print("avg_loss: {:.6f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".\
               format(avg_train_loss, train_output["true_prediction"], train_output["false_prediction"], train_output["accuracy"], \
                     train_output["precision"], train_output["recall"], train_output["f1_score"], train_output["AUC"], train_output["pr_auc"], \
                      utils.format_time(t2-t1)))
        if accelerator.is_main_process:
            gain_1=train_output["GAIN"]["1%"]
            gain_5=train_output["GAIN"]["5%"]
            gain_10=train_output["GAIN"]["10%"]
            with open(os.path.join(os.getcwd(),"metrics_training.txt"),'a') as f:
                f.write(f'{args.model_output_name},{epoch},{avg_train_loss},{train_output["true_prediction"]},{train_output["false_prediction"]},{train_output["accuracy"]},{train_output["precision"]},{train_output["recall"]},{train_output["f1_score"]},{gain_1},{gain_5},{gain_10},{train_output["AUC"]},{train_output["pr_auc"]}\n')    

        t3=time.time()
        
        test_pred,test_target,test_losses=utils.eval_func(test_dataloader,model,accelerator.device)
        avg_test_loss=np.mean(test_losses)
        test_output=utils.model_evaluate(test_target.reshape(-1),test_pred)

        t4=time.time()
        accelerator.print("")
        accelerator.print("==> Running Validation on test set \n")
        accelerator.print("")
        accelerator.print("avg_loss: {:.6f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".\
               format(avg_test_loss, test_output["true_prediction"], test_output["false_prediction"], test_output["accuracy"], \
                     test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"], \
                      utils.format_time(t4-t3)))  

        if accelerator.is_main_process:
            gain_1=test_output["GAIN"]["1%"]
            gain_5=test_output["GAIN"]["5%"]
            gain_10=test_output["GAIN"]["10%"]
            with open(os.path.join(os.getcwd(),"metrics_test.txt"),'a') as f:
                f.write(f'{args.model_output_name},{epoch},{avg_test_loss},{test_output["true_prediction"]},{test_output["false_prediction"]},{test_output["accuracy"]},{test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{gain_1},{gain_5},{gain_10},{test_output["AUC"]},{test_output["pr_auc"]}\n')    

        if accelerator.is_main_process:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
                
        selected_metric=avg_test_loss
        if selected_metric<best_metric:
            best_metric=selected_metric
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                accelerator.print("")
                logger.info(f'Performance improve after epoch: {epoch+1} ... ')
                accelerator.print("")                


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='BERT Model')
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--shuffle_train",  type=bool,default=True,help="shuffle data or not")
    parser.add_argument("--validation_split",  type=float,default=0.2,help="The split ratio for validation dataset")
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=4,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=8,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    #     parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_checkpoint', type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(),"bert_repo"), type=str, help="output folder name")
    parser.add_argument("--model_output_name", default="bert", type=str)
    parser.add_argument("--feature_name", default="TextBody", type=str)
    # parser.add_argument("--frozen_layers", type=int, default=6,help="freeze layers without gradient updates")
    
    args= parser.parse_args()

    # args.model_output_name=f'{args.model_output_name}_{args.truncation_strategy}'
    # args.output_dir=f'{args.output_dir}_{args.truncation_strategy}'
    args.output_dir=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1] + "_repo"
    args.model_output_name=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1]

    seed_everything(args.seed)

    print()
    print(args)
    print()
    
    input_dir="s3://trident-retention-output/"
    train_df=pd.read_csv(os.path.join(input_dir,"train_df.csv"))
    val_df=pd.read_csv(os.path.join(input_dir,"val_df.csv"))
    test_df=pd.read_csv(os.path.join(input_dir,"test_df.csv"))
    
    train_df.drop(['Unnamed: 0'],axis=1,inplace=True)
    val_df.drop(['Unnamed: 0'],axis=1,inplace=True)
    test_df.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    hf_train=Dataset.from_pandas(train_df)
    hf_val=Dataset.from_pandas(val_df)
    hf_test=Dataset.from_pandas(test_df)
    # hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})
    hf_data=concatenate_datasets([hf_train,  hf_val],split="train")
    hf_data=DatasetDict({"train":hf_data, "test":hf_test})

    hf_data=hf_data.filter(lambda x: x[args.feature_name]!=None)
    
    tokenizer=AutoTokenizer.from_pretrained(args.model_checkpoint)

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
    
    # train_data=hf_data['train'].shuffle(seed=101).select(range(800))
    # val_data=hf_data['val'].shuffle(seed=101).select(range(200))
    # test_data=hf_data['test'].shuffle(seed=101).select(range(500))

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

    
    main(args,train_data, test_data,device)
    
    