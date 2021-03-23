#Basic Python and Machine learning libraries
import os, sys, warnings, random, time, re, math, string, demoji, emoji, copy
import pandas as pd
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
import networkx as nx

from string import punctuation
from collections import Counter
from re import search
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams

import spacy
from spacy_langdetect import LanguageDetector

warnings.filterwarnings('ignore')
demoji.download_codes()

#tqdm with pandas
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

# sklearn data science models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.svm import LinearSVC
import xgboost as xgb

from wordsegment import load, segment

def basic_clean(text):
    # text = emoji.demojize(text)
    text = demoji.replace_with_desc(text, sep = " ") #Emoji to text
    text = BeautifulSoup(text, 'lxml').get_text() #links
    text = re.sub('https?://\S+|www\.\S+', '', text) #links
    text = re.sub(' +', ' ', text) 
    text = re.sub("^RT", "", text) #remove RT
    text = re.sub(' +', ' ', text) 
    text = re.sub("^QT", "", text) #remove QT
    text = re.sub(' +', ' ', text) 
    text = re.sub('\n', " ", text) #newlines
    text = re.sub(' +', ' ', text) 
    text = str(text).lower() #makes everything lowercase
    emoji_pattern = re.compile("["
                          "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "\U0001F300-\U0001F5FF"  # symbols & pictographs
                          "\U0001F600-\U0001F64F"  # emoticons
                          "\U0001F680-\U0001F6FF"  # transport & map symbols
                          "\U0001F700-\U0001F77F"  # alchemical symbols
                          "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                          "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                          "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                          "\U0001FA00-\U0001FA6F"  # Chess Symbols
                          "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                          "\U00002702-\U000027B0"  # Dingbats
                          "\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)
    text = re.sub(' +', ' ', text) 
    text = text.replace('@', ' ').replace('#',' ').replace('_',' ') #removes @ # _ from mentions and user-handles, does not remove the handle itself
    return text
  
def camel_case_split(identifier):
  matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
  return " ".join([m.group(0) for m in matches])

def clean_all(text):
    text = basic_clean(text)
    text = camel_case_split(text)

    return text

def remove_duplicates(df, mode='tweet'):
    if mode=='tweet':
        COL_NAME = 'Tweet'
        THRESHOLD = 0.2
    elif mode == 'article':
        COL_NAME = 'Text'
        THRESHOLD = 0.05
    temp_df=copy.deepcopy(df[f'{COL_NAME}'].apply(clean_all))
    labels = temp_df.duplicated()    
    edit =[]
    print("Num of duplicates: ", labels.sum())

    for i in range(len(labels)-1):
        for j in range(i+1,len(labels)):
            if labels.iloc[i] or labels.iloc[j]:
                continue
            dist = Levenshtein.distance(temp_df.iloc[i],temp_df.iloc[j])
            edit.append((dist, i, j))
    dumbs = []
    for a, b, c in edit:
        if a/max(len(temp_df.iloc[b]),len(temp_df.iloc[c])) < THRESHOLD:
            dumbs.append((b,c))
    G = nx.Graph()
    G.add_edges_from(dumbs)
    connected = list(nx.connected_components(G))
    for i,connect in enumerate(connected):
        connect = list(connect)
        for j in range(1,len(connect)):
            labels.iloc[connect[j]] = True
    final  = copy.copy(df[~labels])
    
    return final

if __name__ == '__main__' :
    tweets = pd.read_excel('Development_Data/dev_data_tweet.xlsx')
    tweets.Tweet = tweets.Tweet.apply(basic_clean)
    print(tweets.columns)
    t1 = time.time()
    new_df = remove_duplicates(tweets)
    t2 = time.time()
    print(new_df.shape)
    print(t2-t1)
