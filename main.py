#!/usr/bin/env python3

"""
File name     : main.py
Author        : Ryky Nelson
Created Date  : 10/18/2021
Python Version: Python 3.6.9

Main function: 
- gets & processes the data
- separates the data into the training and test sets
- calls & feeds training data to the perceptron
- measures the training perceptron against (sparred) test data
"""

import pandas as pd

from perceptron import perceptron

if __name__ == "__main__":
    with  open("train.csv", 'r') as tr:
        data = pd.read_csv( tr )

    digit = 1
    data  = data.sample(frac=1, random_state=69).reset_index(drop=True)
    data['label0'] = [ 1 if row == digit else -1 for row in data['label'] ]

    ndata  = len(data)
    ntrain = int(0.80 * ndata)
    nfeat  = len( data.columns ) - 1
    
    training = data.loc[ [*range(ntrain)] ]
    test     = data.loc[ [*range(ntrain,ndata)] ]

    Xtrain = training[ training.columns[1:nfeat] ]
    Ytrain = training[['label0']].values.ravel()

    per = perceptron()
    per.fit(Xtrain, Ytrain)

    Xtest = test[ test.columns[1:nfeat] ]
    Ytest = test[['label0']].values.ravel()

    Ypred = per.predict(Xtest)
    print( 'Accuracy = %3.1f%%' %\
           ( (Ytest == Ypred).sum() * 100 / len(Ytest) ) )
