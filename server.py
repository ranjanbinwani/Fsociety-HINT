# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:30:46 2018

@author: Aayush
"""

from flask import Flask, request
from utils import getFollowerCount, getSeconds, clean_text, clean_light
from flask import jsonify
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
import tweepy 
import pandas as pd
import numpy as np
import csv
from gensim.summarization import summarize

consumer_key = 'HRlTA8OLC6ZmICZIgU21Nn9pX'
consumer_secret = 'U7we8922EU3sgT17BlrYoCPADzhICHIZoN1BFhXHQJ4SwObjJD'
access_key = '944459306119831552-0KrWUnfb4zHMHdaWc8LL6Ej97sAz88j'
access_secret = 'Vs9H82kEnE9YnFGSelYGMCOhwDSQLFnDC67xF4KcxU45z'

auth = tweepy.OAuthHandler('HRlTA8OLC6ZmICZIgU21Nn9pX', 'U7we8922EU3sgT17BlrYoCPADzhICHIZoN1BFhXHQJ4SwObjJD')
auth.set_access_token( '944459306119831552-0KrWUnfb4zHMHdaWc8LL6Ej97sAz88j','Vs9H82kEnE9YnFGSelYGMCOhwDSQLFnDC67xF4KcxU45z')


app = Flask(__name__)

@app.route('/like_pred',methods=['GET', 'POST'])
def getLikeCount():
    if request.method == "POST":
        max_length = 139
        username = request.form.get('username')
        tweet = request.form.get('tweet')
        date = request.form.get('date')
        if username == '' or tweet == '' or date == '':
            return jsonify({"message":"Incomplete Data"})
        followers = getFollowerCount(username)
        seconds = getSeconds(date)
        print (username, tweet, date, followers, seconds)
        with open('word2int.pickle', 'rb') as handle:
            word2int = pickle.load(handle)
        
        X = np.zeros((1, 2), dtype=object)
        X[0, 0] = followers
        X[0, 1] = seconds
        tweet = clean_text(tweet)
        t = []
        _temp = []
        for j in tweet.split():
            _temp.append(word2int[j])
        t.append(_temp)
        t = pad_sequences(t, max_length)
        t = np.array(t)
        
        X_f_test = np.append(X, t, axis = 1)
        loaded_model = pickle.load(open('model_like_pred.sav', 'rb'))
        pi = int(loaded_model.predict(X_f_test))
        print (pi)
        return jsonify({"likes":pi})
        
    else:
        return jsonify({"message":"Invalid Request"})
    
@app.route('/sum',methods=['GET', 'POST'])
def getSummary():
    if request.method == "POST":
        username = request.form.get('username')
        if username == '':
            return jsonify({"message":"Incomplete Data"})
        data = {}
        
        api = tweepy.API(auth)
        user = api.get_user(username)
        count = 0
        for friend in user.friends():
            if count == 5:
                break
            count += 1
            text = scrap(friend.screen_name)
            text = clean_light(text)
            print(text)
            data[str(friend.screen_name)] = str(summarize(text))
        
        return jsonify(data)
    else:
        return jsonify({"message":"Invalid Request"})

def scrap(screen_name):  
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []	
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    count = 0
    while len(new_tweets) > 0:
        print ("getting tweets before %s" % (oldest))
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        if(count == 1):
            break
        count += 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))
    #csvFile = open('tweets_data.csv', 'a')

    # csvWriter = csv.writer(csvFile)
    text=[]
    for tweet in alltweets:
        t=str(tweet.text.encode('utf-8'))
        text.append(t)
        # csvWriter.writerow([screen_name, tweet.created_at , tweet.text.encode('utf-8'), tweet.favorite_count])
        # print (tweet.created_at, tweet.favorite_count , tweet.text)
    # csvFile.close()
    text = ' '.join(text)
    return (text)
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
