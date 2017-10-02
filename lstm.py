# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:11:46 2017

@author: cbasu
"""

import numpy as np

def sigmoid(z, derivative = False):
    
    if derivative:
        
        return sigmoid(z) * 1.0 - sigmoid(z)
    
    return 1.0/(1.0 + np.exp(-z))

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


class LSTM:
    
    def __init__(self, input_size, output_size, state_size, batch_size = 32):
        
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.hidden_states = np.zeros((batch_size, state_size))
        self.internal_memory = np.zeros((batch_size ,state_size))
        self.W_xic = np.random.randn(input_size, state_size) * .01
        self.W_xio = np.random.randn(input_size, state_size) * .01
        self.W_xfg = np.random.randn(input_size, state_size) * .01
        self.W_xog = np.random.randn(input_size, state_size) * .01
        
        self.W_hic = np.random.randn(state_size, state_size) * .01
        self.W_hio = np.random.randn(state_size, state_size) * .01
        self.W_hfg = np.random.randn(state_size, state_size) * .01
        self.W_hog = np.random.randn(state_size, state_size) * .01
        
        self.W_y = np.random.randn(state_size, output_size) * .01
        
        self.b_ic = np.random.randn(state_size)
        self.b_io = np.random.randn(state_size)
        self.b_fg = np.random.randn(state_size)
        self.b_og = np.random.randn(state_size)
        self.b_y = np.random.randn(output_size)
        
        self.io_activations = np.zeros((batch_size, state_size))
        self.ic_activations = np.zeros((batch_size, state_size))
        self.fg_activations = np.zeros((batch_size, state_size))
        self.og_activations = np.zeros((batch_size, state_size))
        
        self.io_preactivations = np.zeros((batch_size, state_size))
        self.ic_preactivations = np.zeros((batch_size, state_size))
        self.fg_preactivations = np.zeros((batch_size, state_size))
        self.og_preactivations = np.zeros((batch_size, state_size))
        
        
        
        
        
        
    def step(self, x, hprev, cprev):
        
        pre_ic = x.dot(self.W_xic) + hprev.dot(self.W_hic) + self.b_ic
        ic = tanh(pre_ic)
        pre_io = x.dot(self.W_xio) + hprev.dot(self.W_hio) + self.b_io
        io = sigmoid(pre_io)
        pre_fg = x.dot(self.W_xfg) + hprev.dot(self.W_hfg) + self.b_fg
        fg = sigmoid(pre_fg)
        pre_og = x.dot(self.W_xog) + hprev.dot(self.W_hog) + self.b_og
        og = sigmoid(pre_og)
        c = np.multiply(ic, io) + np.multiply(cprev, fg)
        h = np.multiply(tanh(c), og)
        
        return h, c, og, fg, io, ic, pre_og, pre_fg, pre_io, pre_ic
    
    
    def forward(self, inputs):
        
        hprev = self.hidden_states[-1]
        cprev = self.internal_memory[-1]
        self.initial_state = hprev
        self.initial_memory = cprev
        for ii, input_ in enumerate(inputs):
            
            hprev, cprev, og, fg, io, ic, pre_og, pre_fg, pre_io, pre_ic = self.step(input_, hprev, cprev)
            self.hidden_states[ii] = hprev
            self.internal_memory[ii] = cprev
            self.og_activations[ii] = og
            self.fg_activations[ii] = fg
            self.io_activations[ii] = io
            self.ic_activations[ii] = ic
            self.og_preactivations[ii] = pre_og
            self.fg_preactivations[ii] = pre_fg
            self.io_preactivations[ii] = pre_io
            self.ic_preactivations[ii] = pre_ic
            
        
        
        logits = np.dot(self.hidden_states, self.W_y) + self.b_y
        
        probabilities = softmax(logits)
        
        return probabilities
    
    
    def loss(self, inputs, probabilities, y):
        
        hidden_ = list(reversed(self.hidden_states))
        losses = -np.log(probabilities[range(len(y)), np.argmax(y, axis = 1)])
        errors = probabilities
        errors[range(len(y)), np.argmax(y, axis = 1)] -= 1
        db_y = np.sum(errors, axis = 0)
        dW_y = np.dot(self.hidden_states.T, errors)
        
        hidden_gradients = []
        
        #compute gradients with respect to the hidden state
        for error in list(reversed(errors)):
            
            error = np.reshape(error, (1, self.output_size))
            hidden_gradient = np.dot(error, self.W_y.T)
            hidden_gradients.insert(0, hidden_gradient)
            
        hidden_gradients = np.reshape(np.array(hidden_gradients), (-1, self.state_size))
        
        #compute gradients with respect to the internal memory and
        #output gate
        
        dog = np.multiply(tanh(self.internal_memory), hidden_gradients)
 
        dog = np.multiply(sigmoid(self.og_preactivations, derivative = True), dog)
        dc = np.multiply(self.og_activations , hidden_gradients)
        dc = np.multiply(dc , tanh(self.internal_memory, derivative = True))
      
        dW_xog = np.zeros(self.W_xog.shape)
        dW_hog = np.zeros(self.W_hog.shape)
        db_og = np.sum(dog, axis = 0)
        
        for ii in range(len(self.hidden_states)):
            
            input_ = np.reshape(list(reversed(inputs))[ii], (1, self.input_size))
            dog_ = np.reshape(list(reversed(dog))[ii], (1, self.state_size))
            
            dW_xog += np.dot(input_.T, dog_)
            
            try:
                state = np.reshape(hidden_[ii + 1], (1, self.state_size))
                
                dW_hog += np.dot(state.T, dog_)
            
            except IndexError:
                
                state = np.reshape(self.initial_state, (1, self.state_size))
                
                dW_hog += np.dot(state.T, dog_)
                
        
        dfg = []
        
        dfg.append(self.initial_memory)
        for i in self.internal_memory[:-1]:
            
            dfg.append(i)
            
        dfg = np.array(dfg)
        
        dio = self.ic_activations
        dic = self.io_activations
        
        #gradients of the activations of the gates
        dfg, dio, dic = np.multiply(dfg, dc), np.multiply(dio, dc), np.multiply(dic, dc)
        
        #gradients of the preactivations of the gates
        
        dfg, dio, dic = np.multiply(dfg, sigmoid(self.fg_preactivations, derivative = True)), np.multiply(dio, sigmoid(self.io_preactivations, derivative = True)), np.multiply(dic, tanh(self.ic_preactivations, derivative = True))
        
        db_fg = np.sum(dfg, axis = 0)
        db_io = np.sum(dio, axis = 0)
        db_ic = np.sum(dic, axis = 0)
        
        dW_xfg = np.zeros(self.W_xfg.shape)
        dW_xio = np.zeros(self.W_xio.shape)
        dW_xic = np.zeros(self.W_xic.shape)
        dW_hfg = np.zeros(self.W_hfg.shape)
        dW_hio = np.zeros(self.W_hio.shape)
        dW_hic = np.zeros(self.W_hic.shape)
        
        for ii in range(len(probabilities)):
            
            input_ = np.reshape(list(reversed(inputs))[ii], (1, self.input_size))
            dfg_ = np.reshape(list(reversed(dfg))[ii], (1, self.state_size))
            dio_ = np.reshape(list(reversed(dio))[ii], (1, self.state_size))
            dic_ = np.reshape(list(reversed(dic))[ii], (1, self.state_size))
            
            dW_xfg += np.dot(input_.T, dfg_)
            dW_xio += np.dot(input_.T, dio_)
            dW_xic += np.dot(input_.T, dic_)
            
            try:
                
                state = np.reshape(hidden_[ii + 1], (1, self.state_size))
                
            except IndexError:
                
                state = np.reshape(self.initial_state, (1, self.state_size))
            
            dW_hfg += np.dot(state.T, dfg_)
            dW_hio += np.dot(state.T, dio_)
            dW_hic += np.dot(state.T, dic_)
        
            
            
        return np.mean(losses), dW_y, dW_hic, dW_hio, dW_hfg, dW_hog, dW_xic, dW_xio, dW_xfg, dW_xog, db_y, db_ic, db_io, db_fg, db_og
            
    def accuracy(self, probabilities, y):
        
        accuracy = np.argmax(probabilities, axis = 1) == np.argmax(y, axis = 1)
        accuracy = np.mean(accuracy)
        return accuracy
    def fit(self, inputs, targets, learning_rate = 1e-3, epochs = 1000):
        
        for epoch in range(epochs):
            
            start = 0
            end = self.batch_size
            
            while start + end < len(inputs):
                
                xbatch, ybatch = inputs[start: end], targets[start: end]
                start += self.batch_size
                end += self.batch_size
                
                probabilities = self.forward(xbatch)
                
                cost, dW_y, dW_hic, dW_hio, dW_hfg, dW_hog, dW_xic, dW_xio, dW_xfg, dW_xog, db_y, db_ic, db_io, db_fg, db_og = self.loss(xbatch, probabilities, ybatch)
                
            
                self.W_y -= learning_rate * dW_y
                self.W_hic -= learning_rate * dW_hic
                self.W_hio -= learning_rate * dW_hio
                self.W_hfg -= learning_rate * dW_hfg
                self.W_hog -= learning_rate * dW_hog
                self.W_xic -= learning_rate * dW_xic
                self.W_xio -= learning_rate * dW_xio
                self.W_xfg -= learning_rate * dW_xfg
                self.W_xog -= learning_rate * dW_xog
                self.b_y -= learning_rate * db_y
                self.b_ic -= learning_rate * db_ic
                self.b_io -= learning_rate * db_io
                self.b_fg -= learning_rate * db_fg
                self.b_og -= learning_rate * db_og
                
                
                if epoch % 100 == 0:
          
                    print(cost)
                    print(self.accuracy(probabilities, targets))
                
        
            
            
            