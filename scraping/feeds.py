import tweepy 
import pandas as pd
import numpy as np
import csv

consumer_key = 'HRlTA8OLC6ZmICZIgU21Nn9pX'
consumer_secret = 'U7we8922EU3sgT17BlrYoCPADzhICHIZoN1BFhXHQJ4SwObjJD'
access_key = '944459306119831552-0KrWUnfb4zHMHdaWc8LL6Ej97sAz88j'
access_secret = 'Vs9H82kEnE9YnFGSelYGMCOhwDSQLFnDC67xF4KcxU45z'

def scrap(screen_name): 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []	
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    	
    while len(new_tweets) > 0:
    	print ("getting tweets before %s" % (oldest))
    	new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    	alltweets.extend(new_tweets)
    	oldest = alltweets[-1].id - 1
    	print ("...%s tweets downloaded so far" % (len(alltweets)))
    csvFile = open('tweets_data.csv', 'a')

    csvWriter = csv.writer(csvFile)
    for tweet in alltweets:
        csvWriter.writerow([screen_name, tweet.created_at , tweet.text.encode('utf-8'), tweet.favorite_count])
        print (tweet.created_at, tweet.favorite_count , tweet.text)
    csvFile.close()

df = pd.read_csv('log.csv')
X = df.iloc[:,0].values
for i in range(len(X)):
        print(X[i])
        scrap(X[i])