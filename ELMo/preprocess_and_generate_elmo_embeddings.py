import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import nltk
from nltk.corpus import stopwords
import re,string,unicodedata
from bs4 import BeautifulSoup
import tensorflow_hub as hub
import tensorflow as tf
import pickle
nltk.download('stopwords')
pd.set_option('display.max_colwidth', 200)

# read data
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")


#Parse dataset -- In total : 28619 samples

# get stopwords
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

#Apply function on headline column
train_data['headline']=train_data['headline'].apply(denoise_text)
test_data['headline']=test_data['headline'].apply(denoise_text)

#lemmatize (normalize) the text by leveraging the popular spaCy library.
# import spaCy's language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

#lemmatize train and test data
train_data['headline'] = lemmatization(train_data['headline'])
test_data['headline'] = lemmatization(test_data['headline'])

# Get pretrained ELMO vectors
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

#Function to get elmo vectors for words in a string
def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

#Break training and test data down as using ELMO embeddings for all data at once will cause memory issues
list_train = [train_data[i:i+100] for i in range(0,train_data.shape[0],100)]
list_test = [test_data[i:i+100] for i in range(0,test_data.shape[0],100)]

# Extract ELMo embeddings
elmo_train = []
elmo_test = []
tr_len = len(list_train)
te_len = len(list_test)

for i,tr in enumerate(list_train):
    elmo_train.append(elmo_vectors(tr['headline']))
    print(f"Processed {i}/{tr_len}")
for i,te in enumerate(list_test):
    elmo_test.append(elmo_vectors(te['headline']))
    print(f"Processed {i}/{te_len}")

# Concatenate data back again
elmo_train_data = np.concatenate(elmo_train, axis = 0)
elmo_test_data = np.concatenate(elmo_test, axis = 0)

with open('../Data/elmo_train_data.pkl','wb') as f:
    pickle.dump(elmo_train_data, f)
    
with open('../Data/elmo_test_data.pkl','wb') as f:
    pickle.dump(elmo_test_data, f)