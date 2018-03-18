# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 02:51:39 2018

@author: Aayush
"""

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import string
from utils_text_gen import *

def clean_light(text):
    words = text.split()
    words = [w for w in words if '@' not in w]
    text = ' '.join(words)
    text = re.sub("b'", " ", text)
    text = re.sub("b'", " ", text)
    text = re.sub("", " ", text)
    text = re.sub("", " ", text)
    text = re.sub("", " ", text)
    text = re.sub("", " ", text)
    text = re.sub("", " ", text)
    text = re.sub("", " ", text)
    words = text.split() 
    #table = str.maketrans('', '', string.punctuation)
    words = [w for w in words if 'https' not in w and 'xf' not in w and 'xe' not in w and 'xc' not in w 
             and 'x9' not in w and '#' not in w]
    #stripped = [w.translate(table) for w in words]
    words = [word.lower() for word in words]
    text = ' '.join(words)
    return text

df = pd.read_csv('text.csv',encoding = "ISO-8859-1")
text = df.iloc[:, 1].values
text = ' '.join(text)
t = clean_light(text)
file_len = len(t)
file = t
len(t)
print_every = 100
n_epochs = 2000
plot_every = 10
n_layers = 1
lr = 0.005
hidden_size = 100
chunk_len = 20
all_characters = string.printable
n_characters = len(all_characters)

def random_training_set(chunk_len):
    st = random.randint(0, file_len - chunk_len)
    fn = st + chunk_len + 1
    chunk = file[st:fn]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
start = time.time()
all_losses = []
criterion = nn.CrossEntropyLoss()


loss_avg = 0

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def save():
    filename = 'generate'
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)
    

#training
print("Training for %d epochs..." % n_epochs)
for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set(chunk_len))
    loss_avg += loss

#saving
save()