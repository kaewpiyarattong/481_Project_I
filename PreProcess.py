from nltk import PorterStemmer, word_tokenize
import json
import pickle

import numpy as np
import pandas as pd
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity

from BM25 import BM25

# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s

lyric_data = pd.read_csv('./data/data.csv')
lyrics = lyric_data['Lyric']
bm25 = BM25()
bm25.fit(lyrics)
pickle.dump(bm25, open('./data/data_bm25.pkl', 'wb'))
