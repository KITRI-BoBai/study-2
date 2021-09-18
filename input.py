import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Load Model
model = load_model('my_model.h5')

# Data Pre-Processing
train = pd.read_csv('./train.csv', encoding = 'utf-8')
test = pd.read_csv('.test.csv', encoding = 'utf-8')
sample_submission = pd.read_csv('./sample_submission.csv', encoding = 'utf-8')

train = train.dropna(how = 'any')
train['data'] = train['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test['data'] = test['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','을']

okt = Okt()

X_input = []
for sentence in input['data']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_input.append(temp_X)# Load Model
model = load_model('my_model.h5')
max_len = 500
tokenizer = Tokenizer()
tokenizer = Tokenizer(vocab_size) 
X_input = tokenizer.texts_to_sequences(X_input)
X_input = pad_sequences(X_input, maxlen = max_len)

predict_x=model.predict(X_input) 
classes_x=np.argmax(predict_x,axis=1)

sample_submission['category'] = classes_x
sample_submission.to_csv('input-result.csv', encoding='utf-8', index=False)
