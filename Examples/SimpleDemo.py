'''
A really simple demo which demonstrates how to instantiate a recurrent convolutional neural networks
model. Here we're just doing a single fold, but typically you'd loop through doing multiple folds.
Always remember... CHANCE ISN'T CHANCE when you're decoding! Generate a null distribution
and compare to that to test for significance. 

'''

import numpy as np
import pandas as pd
from RCNN import RCNN
from RCNN import utils
#from RCNN_Toolbox import Ensembling
from sklearn.metrics import accuracy_score
import sys
import os
sys.setrecursionlimit(1500)

#You want to create variables X formatted as (nTrials x nTimepoints x nElectrodes) and Y formatted as (nTrials x 1)
#We're going to load some sample data from this paper: http://www.sciencedirect.com/science/article/pii/S0893608009001075
print('Downloading Example Data...'),
utils.DownloadExampleData()
print('Done!')
npzfile = np.load('exampledata.npz')
X = npzfile['X']
y = npzfile['y']

Length = X.shape[1]
Chans = X.shape[2]
nbClasses = len(np.unique(y))

model = RCNN.makeModel(Chans, Length, nbClasses, nbRCL=5, nbFilters=32, earlystopping=True, patience=20, filtersize=3, epochs=200, verbose=2)

X_train, y_train, X_test, y_test = utils.train_test_splitter(X, y)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
