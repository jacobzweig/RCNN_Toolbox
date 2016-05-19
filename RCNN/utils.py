'''
Some utility functions.
Very unorganized currently...
'''


import numpy as np
import sys
import os
from sklearn.cross_validation import train_test_split


def train_test_splitter(X,Y, train=None, test=None, test_size=0):
    '''
    Split data into train and test splits
    '''


    if train is not None:
        X_train = X[train]
        X_test = X[test]    
        Y_train = Y[train]
        Y_test = Y[test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size)

    X_train, y_train, X_test, y_test = reshapeData(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test

     
def reshapeData(X_train, y_train, X_test, y_test):

    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    X_train= np.float32(np.transpose(X_train, [0,2,3,1]))
    X_test = np.float32(np.transpose(X_test, [0,2,3,1]))

    y_train =np.squeeze(np.int32(y_train))  
    y_test = np.squeeze(np.int32(y_test))

    return X_train, y_train, X_test, y_test
