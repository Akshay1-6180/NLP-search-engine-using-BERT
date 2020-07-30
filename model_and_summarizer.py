import nltk
import string
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk import tokenize
from sentence_transformers import SentenceTransformer
import string
import pickle
from bs4 import BeautifulSoup
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model_content = T5ForConditionalGeneration.from_pretrained('T5_small')
tokenizer = T5Tokenizer.from_pretrained('T5_small')

def summarizer(data):
  summarized = []
  for i in range(len(data)):
    print(i)
    device = torch.device('cuda')#if u are running in cpu change to cpu
    model_content.to(device)
    text = data['text_ranked'][i]
    preprocess_text = text.strip().replace("\n"," ")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model_content.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summarized.append(output)
  data['text_summary'] = summarized
  return data

def sentence_creator(ranked_content):
  sentence = []
  article = []
  summary = []
  title = []
  url = []
  for i in range(len(ranked_content)):
    text = BeautifulSoup(ranked_content['text'][i], 'html.parser').get_text()
    text = tokenize.sent_tokenize(text)
    sentence.append(ranked_content['title'][i])
    url.append(ranked_content['url'][i])
    summary.append(ranked_content['summarised_text'][i])
    title.append(ranked_content['title'][i])
    article.append(i)
    for j in range(len(text)):
      text[j] = text[j].translate(str.maketrans(" ", " ", string.punctuation))
      text[j] = text[j].lower()
      text[j] = text[j].replace('fm', 'formula milk')
      text[j] = text[j].replace('bm', 'breast milk')
      text[j] = text[j].replace('pls', 'please')
      sentence.append(text[j])
      summary.append(ranked_content['summarised_text'][i])
      article.append(i)
      url.append(ranked_content['url'][i])
      title.append(ranked_content['title'][i])
  sentence_long = []
  article_long = []
  summary_long = []
  title_long = []
  url_long = []
  for i in range(len(sentence)):
    if(len(sentence[i])>=5):
      sentence_long.append(sentence[i])
      article_long.append(article[i])
      summary_long.append(summary[i])
      title_long.append(title[i])
      url_long.append(url[i])
  data = {'article_index': article_long,
          'title' :title_long,
          'snippet': sentence_long,
          'summary' :summary_long,
          'url':url_long}
  return data


content = pd.read_csv('final_content.csv')
print("this will take some time if u are running it for many articles")
content = summarizer(content)
print("running the sentence splitting")
snippet_content = pd.DataFrame(sentence_creator(content), columns = ['article_index','title', 'snippet','summary','url'])
snippet_content.to_excel('snippet_content.xlsx',index=False)

sentence = []
for i in range(len(snippet_content)):
  sentence.append(snippet_content['snippet'][i])

model = SentenceTransformer('model')
sentence_content_embeddings = model.encode(sentence)

with open('sentence_split_encoder_content', 'wb') as f:
    pickle.dump(sentence_content_embeddings, f)


