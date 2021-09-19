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
import csv

# Data Pre-Processing
train = pd.read_csv('./train.csv', encoding = 'utf-8')
test = pd.read_csv('./test.csv', encoding = 'utf-8')
sample_submission = pd.read_csv('./sample_submission.csv', encoding = 'utf-8')

train = train.dropna(how = 'any')
train['data'] = train['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test['data'] = test['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','을']

okt = Okt()

X_train = []
for sentence,i in zip(train['data'],tqdm(range(len(train['data'])))) :
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)

with open('x_train.csv','w') as file:
    write = csv.writer(file)
    write.writerows(X_train)

X_test = []
for sentence in test['data']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = 30000
tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

y_train = to_categorical(train['category'])

# Save as CSV file
with open('x_train_pad.csv','w') as file:
    write = csv.writer(file)
    write.writerows(X_train)

with open('x_test_pad.csv','w') as file:
    write = csv.writer(file)
    write.writerows(X_test)

# Model Defnition
model = Sequential()
model.add(Embedding(vocab_size, 120))
model.add(LSTM(120))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=128, epochs=15)

# Save Model
model.save('my_model.h5')
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)

# Result
sample_submission['category'] = classes_x
sample_submission.to_csv('train-result.csv', encoding='utf-8', index=False)
