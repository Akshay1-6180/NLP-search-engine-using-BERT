from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import scipy
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None
import torch

model = SentenceTransformer('model')

with open('sentence_split_encoder_content', 'rb') as f:
    sentence_embeddings = pickle.load(f)
  
def pre_processing(question):
    def lemmatize_with_pos_tag(sentence):
        tokenized_sentence = TextBlob(sentence)
        tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
        words_and_tags = [(word, tag_dict.get(pos[0], 'n')) for word, pos in tokenized_sentence.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        return " ".join(lemmatized_list)
    question = BeautifulSoup(question, 'html.parser').get_text()
    question = question.lower()
    question = question.replace('fm', 'formula milk')
    question = question.replace('bm', 'breast milk')
    question = question.replace('pls', 'please')
    question = question.translate(str.maketrans(" ", " ", string.punctuation))
    #question = lemmatize_with_pos_tag(question) 
    return question

ranked_content = pd.read_excel('snippet_content.xlsx')


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home_content1.html')


@app.route('/predict',methods=['POST'])
def predict():
  if request.method == 'POST':
    message = request.form['message']
    queries = pre_processing(message)
    queries = [queries]
    
    query_embeddings = model.encode(queries)
    title = []
    score = []
    snippet = []
    summary = []
    url = []
    k = []
    i = 1
    for query, query_embedding in zip(queries, query_embeddings):
      distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
      results = zip(range(len(distances)), distances)
      results = sorted(results, key=lambda x: x[1])
    for idx, distance in results[:100]:
      k.append(i)
      
      if(i==1):
        title.append(ranked_content['title'][idx])
        score.append((1-distance))
        snippet.append(ranked_content['snippet'][idx])
        summary.append(ranked_content['summary'][idx])
        url.append(ranked_content['url'][idx])
      i+=1
      new_title = ranked_content['title'][idx]
      flag = True
      for j in range(len(title)):
        if(new_title==title[j]):
          flag = False
      if(flag):
        title.append(ranked_content['title'][idx])
        score.append((1-distance))
        snippet.append(ranked_content['snippet'][idx]+ " " + ranked_content['snippet'][idx+1])
        summary.append(ranked_content['summary'][idx])
        url.append(ranked_content['url'][idx])

    return render_template('result_content.html',prediction = title[:10],item = k,similarity = score,summary = snippet,website = url)

if __name__ == '__main__':
	app.run()
