import json
import pickle

import nltk
import numpy as np
import pandas as pd
import string
from PreProcess import preProcess

from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask import Flask, request
app = Flask(__name__)

tfidf_vec = pickle.load(open('data/data_tfidf_vec.pkl', 'rb'))
tfidf_X = pickle.load(open('./data/data_tfidf_x.pkl', 'rb'))
tf_x = pickle.load(open('./data/data_tf_X.pkl', 'rb'))
tf_vec = pickle.load(open('./data/data_tf_vec.pkl', 'rb'))
bm25_vec = pickle.load(open('./data/data_bm25.pkl', 'rb'))

def get_and_clean_data():
    df = pd.read_csv('data/lyrics-data.csv')
    df = df[df['Idiom'] == 'ENGLISH']
    df = df.drop_duplicates(subset='SLink')
    description = df['Lyric']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' '*len(string.whitespace), '')))
    df['Lyric'] = cleaned_description
    df.to_csv('data/data.csv')
    return df

def sk_vectorize_tf_idf():
    cleaned_description = get_and_clean_data()
    vectorizer = TfidfVectorizer(preprocessor=preProcess, ngram_range=(1,3))
    X = vectorizer.fit_transform(cleaned_description)
    pickle.dump(vectorizer, open('./data/data_tfidf_vec.pkl', 'wb'))
    pickle.dump(X, open('./data/data_tfidf_x.pkl', 'wb'))

def sk_vectorize_tf():
    cleaned_description = get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess, ngram_range=(1, 3))
    X = vectorizer.fit_transform(cleaned_description)
    X.data = np.log10(X.data + 1)
    pickle.dump(vectorizer, open('./data/data_tf_vec.pkl', 'wb'))
    pickle.dump(X, open('./data/data_tf_X.pkl', 'wb'))

@app.route('/tfidfRanking', methods=['POST', 'GET'])
def tfidf_rank():
    body = request.get_json()
    lyric_data = pd.read_csv('./data/data.csv')
    query_vec = tfidf_vec.transform(body['query'])
    df_cos = cosine_similarity(query_vec,tfidf_X).reshape((-1),)
    tfidf_ranking = pd.DataFrame({'tfidf': list(df_cos), 'Name:':list(lyric_data['ALink']), 'Song:':list(lyric_data['Song']), 'Lyric:':list(lyric_data['Song'])}).nlargest(columns='tf', n=10)
    tfidf_ranking['Rank'] = tfidf_ranking['tfidf'].rank(ascending=False)
    tfidf_ranking = tfidf_ranking.drop(columns='tfidf_ranking',axis=1)
    tfidf_ranking = tfidf_ranking.to_dict('record')
    return tfidf_ranking

@app.route('/tfRanking', methods=['POST', 'GET'])
def tf_query():
    body = request.get_json()
    lyric_data = pd.read_csv('./data/data.csv')
    query_vec = tf_vec.transform(body['query'])
    df_cos = cosine_similarity(query_vec,tf_x).reshape((-1),)
    tf_ranking = pd.DataFrame({'tf': list(df_cos), 'Name:':list(lyric_data['ALink']), 'Song:':list(lyric_data['Song']), 'Lyric:':list(lyric_data['Song'])}).nlargest(columns='tf', n=10)
    tf_ranking['Rank'] = tf_ranking['tfidf'].rank(ascending=False)
    tf_ranking = tf_ranking.to_dict('record')
    return tf_ranking

@app.route('/bm25Ranking', methods=['POST', 'GET'])
def bm25():
    body = request.get_json()
    lyric_data = pd.read_csv('./data/data.csv')
    score = bm25_vec.transform(body['query'])
    bm25_ranking = pd.DataFrame({'tf': list(score), 'Name:': list(lyric_data['ALink']), 'Song:': list(lyric_data['Song']),'Lyric:': list(lyric_data['Lyric'])}).nlargest(columns='tf', n=10)
    bm25_ranking['Rank'] = bm25_ranking['bm25'].rank(ascending=False)
    bm25_ranking = bm25_ranking.to_dict('record')
    return bm25_ranking

# @app.route('/getArtist', methods=['POST'])
# def getArtist():
#     body = request.get_json()
#     data = pd.read_csv('./data/data.csv')
#     result = data['ALink'] == body['Name']
#     result = result.sort_value('Song')
#     return result


if __name__ == "__main__" :
    app.run(debug=True)
