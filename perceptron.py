#!/usr/bin/env python3

"""
File name     : perceptron.py
Author        : Ryky Nelson
Created Date  : 10/18/2021
Python Version: Python 3.6.9

Preceptron class:
train the model, i.e. obtain the weight vector that form the hyperplane that separate the data
into two classes, i.e. the binary classification
"""

import numpy as np

__author__    = "Ryky Nelson"
__copyright__ = "Copyright 2021"

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
        

        
            

        
