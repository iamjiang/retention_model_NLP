import os
import re
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import itertools
import spacy
nlp = spacy.load("en_core_web_md")
from textblob import TextBlob
# python -m textblob.download_corpora
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

all_stopwords_gensim = STOPWORDS.union(set(['thank','thanks', 'you', 'help','questions','a.m.','p.m.','friday','thursday','wednesday','tuesday','monday',\
                                            'askunum','email','askunum.com','unum','askunumunum.com','day','use', 'appreciate','available','mailtoaskunumunum.com',\
                                            'hello','hi','online','?','.','. .','phone','needs','need','let','know','service','information','time','meet','client',\
                                           'team','ask','file','date','opportunity','original','benefit','eastern','specialists','specialist','attached','experienced',\
                                            'benefits insurance','employee','click','organization','httpsbit.lycjrbm',  'received', 'billing', 'manager', 'assist', \
                                            'additional', 'response','vlif']))


def text_preprocess(text, extract_adj=False):
    # lemma = nltk.wordnet.WordNetLemmatizer()
    
    text = str(text)
    
    #remove http links from the email
    
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], '')  
    
    text = re.sub("`", "'", text)
    
    #fix misspelled words

    '''Here we are not actually building any complex function to correct the misspelled words but just checking that each character 
    should occur not more than 2 times in every word. It’s a very basic misspelling check.'''

    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    
    if extract_adj:
        ADJ_word=[]
        doc=nlp(text)
        for token in doc:
            if token.pos_=="ADJ":
                ADJ_word.append(token.text)   
        text=" ".join(ADJ_word)    

    # text = [appos[word] if word in appos else word for word in text.lower().split()]
    # text = " ".join(text)
    
    ### Remove stop word
    text = [i for i in word_tokenize(text) if i not in all_stopwords_gensim]
    text = " ".join(text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    #Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split()]
    text=" ".join(text)
    
    # stem
    # ps = PorterStemmer()
    # text=" ".join(set([ps.stem(w) for w in text.split()]))
    
    return text
    
def textblob_sentiment(text):
    pol_score = TextBlob(text).sentiment.polarity
    if pol_score > 0: 
        return 'positive'
    elif pol_score == 0: 
        return 'neutral'
    else: 
        return 'negative'

def vader_sentiment(text):
    
    senti = SentimentIntensityAnalyzer()
    compound_score = senti.polarity_scores(text)['compound']
    
    # set sentiment 
    if compound_score >= 0.05: 
        return 'positive'
    elif (compound_score > -0.05) and (compound_score < 0.05): 
        return 'neutral'
    else: 
        return 'negative'
    
email_all=load_from_disk(os.path.join(os.getcwd(),"dataset","email_all"))

train_data=email_all['train']
test_data=email_all['test']
train_data.set_format(type="pandas")
df_train=train_data[:]
test_data.set_format(type="pandas")
df_test=test_data[:]

# df_train=df_train.sample(10)
# df_test=df_test.sample(10)

df_train["bag_of_word"]=df_train["Full_TextBody"].progress_apply(text_preprocess)
df_test["bag_of_word"]=df_test["Full_TextBody"].progress_apply(text_preprocess)

df_train["adj_bag_of_word"]=df_train["Full_TextBody"].progress_apply(lambda x: text_preprocess(x, extract_adj=True))
df_test["adj_bag_of_word"]=df_test["Full_TextBody"].progress_apply(lambda x: text_preprocess(x, extract_adj=True))

my_folder="s3://trident-retention-output/"
df_train.to_pickle(os.path.join(my_folder,"df_train"))
df_test.to_pickle(os.path.join(my_folder,"df_test"))
