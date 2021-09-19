import csv
from flask import Flask, render_template, request, redirect, url_for
from flask_restx import Api, Resource
from werkzeug.utils import secure_filename
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

app = Flask(__name__)
api = Api(app)

# count: index
# petition: category
count = -1
petition = -1

# Read: CSV to Dictionary
mydict = {}
with open('./train-result.csv', 'r', encoding='utf-8') as input:
    reader = csv.reader(input)
    mydict = {rows[0]:rows[1] for rows in reader}
    count = len(mydict.keys()) - 1

# Load: Model
model = load_model('my_model.h5')

# Load: Pre-Processed Data
X_train = []
with open('x_train.csv', newline='') as f:
    reader = csv.reader(f)
    X_train.append(list(reader))
X_train = sum(X_train, [])

# /upload: HTML Rendering
@app.route('/upload')
def render_html():
    return render_template('server.html')

# /input: File Upload
@app.route('/input', methods = ['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return redirect(url_for('add_result'))

# /add: Add Input Result to Train Result
@app.route('/add')
def add_result():
    global count
    global petition

    input = pd.read_csv('./input.csv', encoding = 'euc-kr')
    input_submission = pd.read_csv('./input_submission.csv', encoding = 'utf-8')

    input['data'] = input['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','을']

    okt = Okt()

    X_input = []
    for sentence in input['data']:
        temp_X = []
        temp_X = okt.morphs(sentence, stem=True)
        temp_X = [word for word in temp_X if not word in stopwords]
        X_input.append(temp_X)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    vocab_size = 30000
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)

    max_len = 500
    X_input = tokenizer.texts_to_sequences(X_input)
    X_input = pad_sequences(X_input, maxlen = max_len)

    print(X_input)
    predict_x=model.predict(X_input) 
    classes_x=np.argmax(predict_x,axis=1)

    input_submission['category'] = classes_x
    input_submission.to_csv('input-result.csv', encoding='utf-8', index=False)

    with open('./input-result.csv', 'r', encoding='utf-8') as input:
        first_line_flag = False
        reader = csv.reader(input)
        for rows in reader:
            if not first_line_flag:
                first_line_flag = True
                continue
            # number = rows[0]
            petition = rows[1]
        f = open('./train-result.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([count, petition])
        mydict[str(count)] = petition
        count += 1
    
    return redirect(url_for('print_result'))

# /output: Print Result
@app.route('/output')
def print_result():
    return {
        "index": "%s" % str(count - 1),
        "category": "%s" % mydict[str(count - 1)]
    }

# /api/{index}: Providing API
@api.route('/api/<string:index>')
class API(Resource):
    def get(self, index):
        return {
            "index": "%s" % index,
            "category": "%s" % mydict[index]
        }

# main: Run server on localhost 5000 port
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)