from mysql.connector import Error
import mysql.connector
import nltk
import string
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import string
import torch
import pickle
from bs4 import BeautifulSoup
from gensim.summarization.summarizer import summarize
nltk.download('averaged_perceptron_tagger')
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
nltk.download('wordnet')

def word_count_creater(data):
  words = []
  for i in range(len(data)):
    words.append(len(data['cleaned_text'][i].split()))
  data['word_count'] = words
  data = data[data['word_count'] >= 100]
  data.reset_index(drop=True, inplace=True)
  return data

def text_rank(data):
  texts = []
  for i in range(len(data)):
    text = data['text'][i]
    text = text.replace('.', '\n')
    text = text.replace('?', '\n')
    rm_html = BeautifulSoup(text, 'html.parser').get_text()
    texts.append(summarize(rm_html,word_count = 512))
  data['text_ranked'] = texts
  return data

def pre_processing(question):
    def lemmatize_with_pos_tag(sentence):
        tokenized_sentence = TextBlob(sentence)
        tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
        words_and_tags = [(word, tag_dict.get(pos[0], 'n')) for word, pos in tokenized_sentence.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        return " ".join(lemmatized_list)
    question = BeautifulSoup(question, 'html.parser').get_text()
    question = question.lower()
    question.translate(str.maketrans(" ", " ", string.punctuation))
    question = lemmatize_with_pos_tag(question)
    return question

def get_content_connection():
    connection = None
    try:
        connection = mysql.connector.connect(host='localhost',
                                                  database='content',
                                                  user='root')
        if connection.is_connected():
            print('Connection made successfully')
    except Error as e:
        print("Error while connecting to MySQL", e)
    return connection

def get_content_data(connection):
    data_frame = None
    try:
        if connection.is_connected:
            sql_query = "SELECT title, text, url, age_group_min, age_group_max, content_type" \
                        " FROM content.article"
            data_frame = pd.read_sql_query(sql_query, connection)
    except Error as e:
        print("There was an error", e)

    if connection.is_connected():
        connection.close()
        print('Connection closed successfully')
    return data_frame

content = get_content_data(get_content_connection())
content['text'] = content['text'].fillna('0')
content = content[content['text'] != '0']
content.reset_index(drop=True, inplace=True)
cleaned_content = []
k = content['text'].values
for i in range(len(k)):
  cleaned_content.append(pre_processing(k[i]))
content['cleaned_text'] = cleaned_content  
content = word_count_creater(content)
content = text_rank(content)
content['text_ranked'] = content['text_ranked'].fillna('0')
content = content[content['text_ranked'] != '0']
content.reset_index(drop=True, inplace=True)
content.to_csv('final_content.csv')
