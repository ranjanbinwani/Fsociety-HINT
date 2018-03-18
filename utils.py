# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:37:04 2018

@author: Aayush
"""

from bs4 import BeautifulSoup
import requests
from datetime import datetime
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from gensim.summarization import summarize

def getFollowerCount(username):
	url = 'https://www.twitter.com/'+ username
	r = requests.get(url)
	soup = BeautifulSoup(r.content, 'lxml')
	f = soup.find('li', class_="ProfileNav-item--followers")
	title = f.find('a')['title']

	num_followers = int(title.split(' ')[0].replace(',',''))
	return (num_followers)

def getSeconds(date_time):
    st = str(str(date_time).split(' ')[0])
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
    return tdelta.total_seconds()

def clean_text(text):
    words = text.split()
    words = [w for w in words if '@' not in w]
    text = ' '.join(words)
    text = re.sub("b'", " ", text)
    text = re.sub("'", " ", text)
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w for w in words if 'https' not in w and 'xf' not in w and 'xe' not in w and 'xc' not in w 
             and 'x9' not in w and '#' not in w]
    stripped = [w.translate(table) for w in words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in stripped]
    words = [word.lower() for word in stemmed]
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    text = ' '.join(words)
    return text

def clean_light(text):
    words = text.split()
    words = [w for w in words if '@' not in w]
    text = ' '.join(words)
    text = re.sub("b'", " ", text)
    text = re.sub("'", " ", text)
    words = text.split() 
    #table = str.maketrans('', '', string.punctuation)
    words = [w for w in words if 'https' not in w and 'xf' not in w and 'xe' not in w and 'xc' not in w 
             and 'x9' not in w and '#' not in w]
    #stripped = [w.translate(table) for w in words]
    words = [word.lower() for word in words]
    text = ' '.join(words)
    return text

def getSummary(text):
    data = str(summarize(text))
    return data
    