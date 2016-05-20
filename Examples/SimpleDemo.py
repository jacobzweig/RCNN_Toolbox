'''
A really simple demo which demonstrates how to instantiate a recurrent convolutional neural networks
model. Here we're just doing a single fold, but typically you'd loop through doing multiple folds.

'''

import numpy as np
import pandas as pd
from RCNN import ModelMaker
from RCNN import utils
#from RCNN_Toolbox import Ensembling
import sys
sys.setrecursionlimit(1500)

#You want to create variables X formatted as (nTrials x nTimepoints x nElectrodes) and Y formatted as (nTrials x 1)
X = []
y = []

Chans = X.shape[2]
Length = X.shape[1]
nbClasses = len(np.unique(y))

model = ModelMaker.init(Chans, Length, nbClasses, nbRCL=5, nbFilters=32, earlystopping=True, patience=20, filtersize=3, epochs=200)

X_train, y_train, X_test, y_test = utils.train_test_splitter(X, Y)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
