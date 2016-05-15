'''
A really simple demo written in a few mins...
More to come soon!
Contact me for help or more info.
'''

import numpy as np
import pandas as pd

#load if not yet installed as a package
from RCNN_Toolbox import RCNN
from RCNN_Toolbox import utils
#from RCNN_Toolbox import Ensembling

import sys
sys.setrecursionlimit(1500)


#You want to create variables X formatted as (nTrials x nTimepoints x nElectrodes) and Y formatted as (nTrials x 1)
X = []
y = []

model = RCNN.init(Chans, Length, nbClasses, nbRCL=6, nbFilters=32, earlystopping=True, patience=20, filtersize=3, epochs=200)

X_train, y_train, X_test, y_test = utils.SplitData(X, Y)
model.fit(X_train, y_train)