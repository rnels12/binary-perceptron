#!/usr/bin/env python3

"""
Created on 10/18/2021

@author: Ryky nelson

Preceptron class:
trains the model, i.e. obtains the weight vector 
that forms the hyperplane separating the data
into two classes, i.e. the binary classification
"""

import numpy as np

class perceptron:
    def __init__(self):
        self.w = np.array([])

    def fit(self, x, y):
        nrow, col = x.shape
        self.xtrain = np.concatenate( (  np.array( x ), np.ones((nrow,1)) ), axis=1 )
        self.ytrain = np.array( y )

        self.w = np.zeros( col + 1 )

        for index, iy in enumerate(self.ytrain):
            dis = iy * np.sign( np.dot( self.w, self.xtrain[index] ) )
            if dis <= 0: self.w += (iy * self.xtrain[index] )
                
    def predict(self, x):
        nrow, col = x.shape
        self.xtest = np.concatenate( (  np.array( x ), np.ones((nrow,1)) ), axis=1 )
        return np.sign( np.dot( self.xtest, self.w.T ) )
        
