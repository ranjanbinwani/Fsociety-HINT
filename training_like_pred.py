# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 08:20:11 2018

@author: Aayush
"""

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import nltk
import pickle
from utils import getFollowerCount, getSeconds
nltk.download('stopwords')
def clean_text(text):
    words = text.split()
    words = [w for w in words if '@' not in w]
    text = ' '.join(words)
    text = re.sub("b'", " ", text)
    text = re.sub("'", " ", text)
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w for w in words if 'https' not in w and 'xf' not in w and 'xe' not in w and 'xc' not in w]
    stripped = [w.translate(table) for w in words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in stripped]
    words = [word.lower() for word in stemmed]
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    text = ' '.join(words)
    return text

def model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=input_length))
    model.add(Flatten())
    model.add(Dense(128,kernel_initializer='normal', activation='relu'))
    model.add(Dense(128,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    return model



df = pd.read_csv('tweets_data.csv')
text = df.iloc[:,2].values
X = df.iloc[:, 0:2].values
y = df.iloc[:,-1].values
k=0
for i in range(len(text)):
    try:
        text[k] = clean_text(text[i])
        X[k] = X[i]
        y[k] = y[i]
        k += 1
    except:
        pass

x = np.zeros((k,2), dtype=object)
yi = np.zeros((k,))
t = np.zeros((k), dtype=object)
for i in range(k):
    x[i] = X[i]
    yi[i] = y[i]
    t[i] = text[i]
X = x
y = yi
text = t


'''mean = 0
for i in range(k):
    mean += y[i]
mean /= k
p=0
for i in range(k):
    if y[i]==0:
        y[i] = int(mean / 10)
 '''       
max_length = -1
for i in range(k):
    max_length = max(max_length, len(text[i]))

for i in range(k):
    if len(text[i]) == max_length:
        print (text[i])

word2int = {}
int2word = {}
words = set()
for i in range(len(text)):
    _t = (sorted(text[i].split()))
    for j in _t:
        words.add(j)
words = list(sorted(words))
vocab_size = len(words)

for i in range(len(words)):
    word2int[words[i]] = i
    int2word[i] = words[i]
    
with open('word2int.pickle', 'wb') as handle:
    pickle.dump(word2int, handle, protocol=pickle.HIGHEST_PROTOCOL)

t = []
for i in range(len(text)):
    _temp = []
    for j in text[i].split():
        _temp.append(word2int[j])
    t.append(_temp)
        
t = pad_sequences(t, max_length)
t = np.array(t)

X_final = np.append(X, t, axis = 1)
#X_final = X_final[:,1:]

df_new = pd.read_csv('log.csv')
data = str(df_new.iloc[:,0].values).split(' ')[4]

X_final[:,0] = data

from datetime import datetime
for i in range(len(X_final)):
    current = np.datetime64(datetime.now())
    #print(X_final[i,0])
    st = str(str(X_final[i,1]).split(' ')[0])
    fn = str(np.datetime64('today'))
    try:
        s = datetime.strptime(str(st), '%d-%m-%Y') 
    except:
        s = datetime.strptime(str(st), '%Y-%m-%d') 
    try:
        f = datetime.strptime(str(fn), '%Y-%m-%d')
    except:
        f = datetime.strptime(str(fn), '%d-%m-%Y')
    tdelta = f - s  
    X_final[i,1] = tdelta.total_seconds()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size = 0.2)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train, y_train)

# saving model
import pickle
filename = 'model_like_pred.sav'
pickle.dump(reg, open(filename, 'wb'))

pred = reg.predict(X_test)

model = model(vocab_size, max_length+2)
model.fit(X_train, y_train, epochs = 20)


# TESTING

username = "realdonaldtrump"
tweet = "It's a wonderful day. People are mad"
date = "2018-03-19"
followers = getFollowerCount(username)
seconds = getSeconds(date)

X = np.zeros((1, 2), dtype=object)
X[0, 0] = followers
X[0, 1] = seconds
tweet = clean_text(tweet)
t = []
_temp = []
for j in tweet.split():
    if j in word2int:
        _temp.append(word2int[j])
t.append(_temp)
t = pad_sequences(t, max_length)
t = np.array(t)

X_f_test = np.append(X, t, axis = 1)

pi = int(reg.predict(X_f_test))
print (pi)
        


