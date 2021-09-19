## study-2: REST API Server using Flask

### Prerequisite
```sh
$ pip3 install flask
$ pip3 install flask-restx
$ pip3 install konlpy
$ pip3 install tqdm
$ pip3 install tensorflow
$ pip3 install pandas
```
```sh
$ sudo apt-get update
$ sudo apt-get install g++ openjdk-8-jdk python-dev python3-dev
$ export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
$ export PATH="$PATH:$JAVA_HOME/bin"
```

### Code Description: model.py
```py
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

```

### Run API Server
```sh
$ python server.py
```

### Code Description: server.py
```py
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

# /input: input.csv File Upload
@app.route('/input', methods = ['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return redirect(url_for('add_result'))

# /add: Predict input-result.csv & Add input result to train-result.csv
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
```
### Result
> ![input](https://user-images.githubusercontent.com/20378368/133913237-c0b9a392-018a-4776-b2f9-80dffdabafee.png)  
> ![api](https://user-images.githubusercontent.com/20378368/133913403-85a7e53f-d966-425a-9464-ca3dd5c345c4.png)