# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:36:47 2017

@author: cbasu
"""
import numpy as np

def tanh(z, derivative = False):
    
    if derivative:
        
        return 1.0 - np.tanh(z)**2
    
    return np.tanh(z)

def softmax(x):
    '''
    Accepts a NxD matrix and returns an NxD matrix
    which can be interpreted as probabilities.
    '''
    x = x - np.max(x, axis = 1, keepdims = True)
    x = np.exp(x)
    return x/np.sum(x, axis = 1, keepdims = True)

class Rnn:
    
    
    def __init__(self, input_size, output_size, batch_size, hidden_size):
        
        self.input_size = input_size
        self.target_size = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_preactivations = np.zeros((batch_size, hidden_size))
        self.hidden_states = np.zeros((batch_size, hidden_size))
        
        self.Wxh = np.random.randn(input_size, hidden_size) * .01
        self.Whh = np.random.randn(hidden_size, hidden_size) * .01
        self.bhh = np.random.randn(hidden_size)
        
        self.Why = np.random.randn(hidden_size, output_size) * .01
        self.bhy = np.random.randn(output_size)
        
    def step(self, x, hprev):
        
        preactivation = x.dot(self.Wxh) + hprev.dot(self.Whh) + self.bhh
        h = tanh(preactivation)
        
        y = h.dot(self.Why) + self.bhy
        
        return preactivation, h, y

    def forward(self, inputs):
        
        logits = np.zeros((len(inputs), self.target_size))
        hprev = self.hidden_states[-1]
        self.initial_state = hprev
        for ii, _input in enumerate(inputs):
            
            preactivation, hprev, y = self.step(_input, hprev)
            self.hidden_states[ii] = hprev
            self.hidden_preactivations[ii] = preactivation
            logits[ii] = y
        
        logits =  logits - np.max(logits, axis = 1, keepdims = True)
        probabilities = softmax(logits)
        return probabilities
    

    
    def loss(self, inputs, probabilities, y):
        
        predicted = probabilities[range(len(probabilities)), np.argmax(y, axis = 1)]
        losses = -np.log(predicted)
        errors = probabilities
        
        errors[range(len(probabilities)), np.argmax(y, axis = 1)] -= 1
        dWhy = np.zeros(self.Why.shape)
        dWxh = np.zeros(self.Wxh.shape)
        dWhh = np.zeros(self.Whh.shape)
        dbhy = np.sum(errors, axis = 0)
        activation_gradients = []
        
        for ii, error in enumerate(reversed(errors)):
            hidden_state = np.reshape(list(reversed(self.hidden_states))[ii], (1 , self.hidden_size))
            error = np.reshape(error, (1, self.target_size))
            dWhy+= np.dot(hidden_state.T, error)
        
        for ii, error in enumerate(reversed(self.hidden_preactivations)):
            activation_gradient = np.dot(list(reversed(errors))[ii], self.Why.T)
            activation_gradients.insert(0, activation_gradient)
            
        activation_gradients = np.array(activation_gradients)
        preactivation_gradients = np.multiply(tanh(self.hidden_preactivations, derivative = True), 
                         activation_gradients)
        
        dbhh = np.sum(preactivation_gradients, axis = 0)
        
        for ii, _input in enumerate(reversed(inputs)):
            _input = np.reshape(_input, (1, self.input_size))
            preactivation_gradient = list(reversed(preactivation_gradients))[ii]
            preactivation_gradient = np.reshape(preactivation_gradient, (1, self.hidden_size))
            dWxh += np.dot(_input.T, preactivation_gradient)
        
        for ii, state in enumerate(reversed(self.hidden_states)):
            
            try:
                dWhh += np.dot(list(reversed(self.hidden_states))[ii + 1].T, 
                               list(reversed(preactivation_gradients))[ii])
            except IndexError:
                
                dWhh += np.dot(self.initial_state.T, preactivation_gradients[ii])
            
        return np.mean(losses), dWxh, dWhh, dWhy, dbhh, dbhy
    
    def fit(self, inputs, targets, learning_rate = 1e-2, epochs = 50):
        
        
        for epoch in range(epochs):
            
            start = 0
            end = self.batch_size
            
            while start + end < len(inputs):
                
                xbatch, ybatch = inputs[start: end], targets[start: end]
                start += self.batch_size
                end += self.batch_size
                
                probabilities = self.forward(xbatch)
                loss, dWxh, dWhh, dWhy, dbhh, dbhy = self.loss(xbatch, probabilities, ybatch)
                
            
                self.Wxh -= learning_rate * dWxh
                self.Whh -= learning_rate * dWhh
                self.Why -= learning_rate * dWhy
                self.bhh -= learning_rate * dbhh
                self.bhy -= learning_rate * dbhy
                
            if epoch % 100 == 0:
            
                print(loss)
                
        
            
            
        
        
        
            

        
        