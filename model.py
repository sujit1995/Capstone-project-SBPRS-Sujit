# Importing Libraries
from nltk.corpus.reader import reviews
import numpy as np
from numpy.core.fromnumeric import prod
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
import pickle as pk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from nltk.corpus import stopwords

# load the pickle files 

count_vector = pk.load(open('count_vector.pkl','rb'))            # Count Vectorizer
tfidf_transformer = pk.load(open('tfidf_transformer.pkl','rb')) # TFIDF Transformer
model = pk.load(open('model.pkl','rb'))                          # Classification Model
recommend_matrix = pk.load(open('user_final_rating.pkl','rb'))   # User-User Recommendation System 

nlp = en_core_web_sm.load()

product_df = pd.read_csv(r'sample30.csv')

def text_preprocessing (text):
    cleaned_text = clean_text(text)
    lemmatized_text = lemmatize(cleaned_text) 
    lemmatized_text = lemmatized_text.replace('-PRON-','')
    return lemmatized_text

# cleaning the text using re library 
def clean_text(text):
  text = text.lower()
  text = re.sub('[0-9]','',text)
  text = re.sub(r'[^a-zA-Z\s]','',text)
  return text 

#lemmatizing the text using Spacy library
def lemmatize(text):
    sent = nlp(text)
    sentence = [token.lemma_ for token in sent if token not in set(stopwords.words('english'))]
    return " ".join(sentence)

#predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

#Recommend the products based on the sentiment from model
def recommend_products(user_name):
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name','reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].apply(text_preprocessing)
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df
    

def top5_products(df):
    total_product=df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name','predict_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['reviews_text'],on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    return merge_df[merge_df['predict_sentiment'] ==  1][:5]