# math and parse
import numpy as np
import pandas as pd
import json
#visualisation
import seaborn as sns
# plotting
import matplotlib.pyplot as plt
#natural language toolkit
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
#from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from gensim.models import Word2Vec
#import torchtext
#from torchtext.data import get_tokenizer
from nltk.tokenize import word_tokenize
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator 
#from torchnlp.encoders.text import StaticTokenizerEncoder,stack_and_pad_tensors,pad_tensor
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')
#import keras
#from keras.models import Sequential
#from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
#import tensorflow as tf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
#loading dataset 
data = pd.read_json("./Sarcasm_Headlines_Dataset_v2.json", lines=True)

#deleting article_link column
del data['article_link']

# number of non sarcastic headlines
#print (data.shape[0]-data.is_sarcastic.sum())

# number of sarcastic headlines
#print(data.is_sarcastic.sum())

# remove stopwords as they don't add any meanings to the sentece

stop = set(stopwords.words('English'))
punctuation = list (string.punctuation)
stop.update (punctuation)

# stopword and punctuation in English language
#print (stop)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

def denoise_data (row):
    row = remove_stopwords(row)
    return row


data ['headline'] = data['headline'].apply (denoise_data)
#print (data.head())

# wordcloud for sarcastic text
'''
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(data[data.is_sarcastic == 1].headline))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Wordcloud for sarcastic headlines')
plt.savefig('Wordcloud for sarcastic headlines.png')

#wordcloud for non sarcastic text
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(data[data.is_sarcastic == 0].headline))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Wordcloud for non sarcastic headlines')
plt.savefig('Wordcloud for non sarcastic headlines.png')
'''
train = pd.read_csv("./train.csv")
#train['headline'].dropna(inplace=True)
test = pd.read_csv("./test.csv")
# Use gensim for modeling
# gensim is an open source library for unsupervised topic modeling 


tokens = [word_tokenize(sentences) for sentences in train]    
    
EMBEDDING_DIM = 200
#Creating Word Vectors by Word2Vec Method from genism library
w2v_model = Word2Vec(tokens , vector_size=EMBEDDING_DIM , window = 5 , min_count = 1)

#building vocabulary for training 
w2v_model.build_vocab(tokens)
print("\n Training the word2vec model...\n")
w2v_model.train(tokens, total_examples = len(tokens), epochs = 4000)


#max_dataset_size = len(w2v_model.wv.syn0)
