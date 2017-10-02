# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:23:33 2017

@author: cbasu
"""
from preprocess import build_dataset, get_corpus, make_mappings
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description = 'Trains a LSTM on Text provided.')
parser.add_argument('file_path', metavar = 'F', type=str, help = 'A path to a text file.')
args = parser.parse_args()
corpus = get_corpus(args.file_path)
inputs, outputs = build_dataset(corpus)
word_to_id, id_to_word = make_mappings(corpus)

from lstm import LSTM

clf = LSTM(inputs.shape[1], outputs.shape[1], 250, 128)



for i in range(10000):
    clf.fit(inputs, outputs, learning_rate = .001, epochs = 1)
    
    generated = []
    index = np.random.randint(len(inputs))
    init = inputs[index]
    hprev, cprev = clf.hidden_states[-1], clf.internal_memory[-1]
    
    generated.append(id_to_word[np.argmax(init)])
    
    for ii in range(500):
        
        states = clf.step(init, hprev, cprev)
        hprev = states[0]
        cprev = states[1]
        
        predicted = hprev.dot(clf.W_y) + clf.b_y
        init = np.zeros(len(predicted)) 
        init[np.argmax(predicted)] = 1
        generated.append(id_to_word[np.argmax(predicted)])
        
    text = ''.join(generated)
    
    if not os.path.exists('text'):
        
        os.mkdir('text')
        
    with open('text/{0}.txt'.format(i), 'w+') as f:
        
        f.write(text)
    
    print(''.join(generated))
        
   
    


    


        