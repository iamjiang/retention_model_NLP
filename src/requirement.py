import os

os.system("pip install awscli")
os.system("pip install boto3")
os.system("pip install s3fs")

os.system("pip install gensim")
os.system("pip install spacy")
os.system("python -m spacy download en_core_web_md")
os.system("pip install torchtext==0.8")
os.system("pip install nltk")
os.system("python -m nltk.downloader stopwords")
os.system("pip install -U pip")
os.system("pip install  transformers==4.6.1")
os.system("pip install -U tensorflow")
os.system("pip install -U sentencepiece")
os.system("pip install -U pytorch-lightning")
os.system("pip install -U seaborn")
os.system("pip install -U tqdm")
os.system("pip install torch==1.9.1 torchvision -f https://download.pytorch.org/whl/cu111/torch_stable.html")
os.system("pip install -U rouge_score")
os.system("pip install -U datasets") 
os.system("pip install -U GPUtil")
os.system("pip install -U accelerate")

import transformers
import torch
print("torch version is {}".format(torch.__version__))
print("Transformers version is {}".format(transformers.__version__))

