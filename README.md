# Machine Learning models to predict customers retention rate
(1) Decision Tree based model(Light Gradient Boosting) to predict unum clients' churn probability.  The jupyter-notebooks are stored in the **structure_model** folder <br/>
<br/>
(2) NLP model to predict unum clients' churn probability.  A wide range of NLP models like bag-of-word model(TF-IDF, CNN), pretrained language model(distillbert,  bert, roberta, longformer) were explored to detected the sentiment of email textbody and use this sentiment analysis to predict clients' churn decision. All of the codes in this part are stored in the **src** folder <br/>
<br/>
(3)  NLP model to predict the multiple classes of email subtype. The pretrained language model like Bert, Roberta model were explored to predict the subtype of email textbody. All codes are stored in the **multi-class** folder <br/>
<br/>
(4) GPT3 model to generate a complete sentence from a keyword.  All codes are stored in **auto-complete-sentence** folder
<br/>
(5) prompt-based zero shot learning to predict sentiment polarity of emails.  All codes are stored in **zero-shot-learning** folder
<br/>
## NLP model to predict churn probability

#### requirement
Huggingface transformers and torch are required to run the codes 

```python
pip install -r requirement.txt
```
#### How to use 
step 1: preprocess email textbody to remove layout/nosiy information from email textbody  
```python 
python 01_text_data_preprocessing.py
```
step 2: merge churn label data to email text data
```python 
python 02_churn_label_data.py
python 02_text+label.py
```
step 3: create training and test data for model training and inference
```python 
python 03_train_test_data.py
```
Three version of email textbody are created: <br/> 
(1) Full_textbody includes client and unum representative email text <br/>
(2) Client_textboday includes client email text only <br/>
(3) Latest_TextBody only include the latest recent email text before the churn decision.<br/>

- CNN model : <br/>
```python
CUDA_VISIBLE_DEVICES=1 python CNN.py \
--gpu 1 \
--loss_weight \
--feature_name Full_TextBody \
--train_batch_size 32 \
--test_batch_size 32 \
--gradient_accumulation_steps 4 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 2e-5 \
--weight_decay 2e-5 \
--keep_probab 0.4 \
--mode static \
--is_pretrain_embedding \
--kernel_heights 3 4 5 \
--out_channels 100 \
--embedding_length 300 
```

- TF-IDF model : <br/>
```python
CUDA_VISIBLE_DEVICES=0 python TF-IDF.py \
--gpu 0 \
--loss_weight \
--feature_name Full_TextBody \
--train_batch_size 256 \
--test_batch_size 256 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 5e-5 \
--weight_decay 1e-3 \
--keep_probab 0.5 \
--max_features 1000
```

- bert-base model : <br/>
```python
CUDA_VISIBLE_DEVICES=3 python bert_base.py \
--feature_name Full_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy tail \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--data Full_TextBody_truncation_tail_bert 
```

- longformer model : <br/>
```python
CUDA_VISIBLE_DEVICES=0 python longformer.py \
--feature_name Full_TextBody  \
--gpus 0 \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--lr 3e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--max_length 4096 \
--data Full_TextBody_truncation_tail_longformer \
--frozen_layers 6
```

## NLP model to predict the multiple classes of email subtype (multi-class classification)

#### requirement
Huggingface transformers and torch are required to run the codes 

```python
pip install -r requirement.txt
```

#### How to use 
step 1: preprocess email textbody to remove layout/nosiy information from email textbody 
```python 
python 01_text_data_preprocessing.py
```
step 2: concatenate email text data per each subtype
```python 
python 02_text_concatenate.py
```
step 3: create training and test data for model training and inference
```python 
python 03_train_test_data.py
```
- roberta-large model : <br/>
```python
CUDA_VISIBLE_DEVICES=1 python bert.py \
--feature_name TextBody  \
--gpus 1 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy head \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--model_checkpoint roberta-large
```
The evaluation of model perfomrance for multi-class classification problem can be found in the notebook **model_performance.ipynb**

## GPT3 model to generate a complete sentence from keywords

#### requirement
OpenAI GPT3 API is required to implement text generation. https://beta.openai.com/docs/api-reference/introduction

```python
pip install openai
```

#### How to use 
Since the current OpenAI account is only for free trial usage, there are some rate limits to use GPT3 model per every minute. Therefore the data is split into 20 chunks and the code is executed sequentially.  In the end, all outputs are combined together to get final results
```python
python GPT3_complete_sentence.py  --idx 0
python GPT3_complete_sentence.py  --idx 1
python GPT3_complete_sentence.py  --idx 2
..........
python GPT3_complete_sentence.py  --idx 19
```

The main function for gpt3 in-context learning are as followings:

```python
def get_gpt3_complete(keyword,max_tokens=15,temperature=0):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=[f"given the key words below, generate a medical related only sentence ### \
      key words: intractable back pain -> sentence: the person has intractable back pain ***  \
      key words: at high risk -> sentence:  the person's condition is at high risk  *** \
      key words: 10 pain -> sentence:  the person has a rating of 10 pain  *** \
      key words: no change -> sentence:  the person's condition has no change *** \
      key words: pain is well controlled -> sentence:  the person control his pain ver well *** \
      key words: a rating of -> sentence:  the person has a rating of 10 pain level  *** \
      key words: good progress -> sentence:  the person has shown good progress in his condition *** \
      key words: {keyword} -> sentence: \
      "],
      temperature=0,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n","<|endoftext|>"]
    )
    return response
```

