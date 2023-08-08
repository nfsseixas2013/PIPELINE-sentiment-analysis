import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup



def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet,'lxml').get_text()
    tweet = re.sub(r"@[A-Za-z0-9]+",' ',tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+",' ', tweet)
    tweet = re.sub(r"[^a-zA-Z.!?']", " ", tweet)
    tweet = re.sub(r" +", " ", tweet)
    return tweet


cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
data =  pd.read_csv('/home/nilton/Arquivos/Softplan/Tutorials/Curso_BERT_Jonathan/bert_data/Base de dados sentimentos/training.1600000.processed.noemoticon.csv',
                header = None,
                names = cols,
                engine='python',
                encoding='latin-1')
data.drop(['id', 'date', 'query', 'user'],
        axis = 1, inplace = True)


data_clean = [clean_tweet(tweet) for tweet in data.text]
data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

dataframe_saida = pd.DataFrame({'data_clean': data_clean, 'data_labels': data_labels})
dataframe_saida.to_csv('dataset.csv', index=False)

