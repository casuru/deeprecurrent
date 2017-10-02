# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:12:11 2017

@author: cbasu
"""
import numpy as np

def get_corpus(path):
    
    with open(path) as f:
        
        text = f.read()
        text = text
        
    return text

def make_mappings(text):
    
    word_to_id = dict((w, i) for i, w in enumerate(set(text)))
    id_to_word = dict((i, w) for w, i in word_to_id.items())
    
    return word_to_id, id_to_word

def to_one_hot(x, k):
    
    one_hot = np.zeros(k)
    one_hot[x] = 1
    
    return one_hot


    
    word_to_id, id_to_word = make_mappings(corpus)

def build_dataset(corpus):
    char_size = len(set(corpus))
    word_to_id, id_to_word = make_mappings(corpus)
    inputs = corpus[:-1]
    targets = corpus[1:]
    inputs = [word_to_id[char] for char in inputs]
    targets = [word_to_id[char] for char in targets]
    
    inputs = [to_one_hot(char, char_size) for char in inputs]
    targets = [to_one_hot(char, char_size) for char in targets]
    
    return np.array(inputs), np.array(targets)


if __name__ == "__main__":
    corpus = get_corpus('corpus.txt')
    word_to_id, id_to_word = make_mappings(corpus)
    inputs, targets = build_dataset(corpus) 


