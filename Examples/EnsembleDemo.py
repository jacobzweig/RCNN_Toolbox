'''
Demonstration of how to use the model ensembling functions
'''

import numpy as np
import pandas as pd
from RCNN_Toolbox import RCNN
from RCNN_Toolbox import utils
from RCNN_Toolbox.Ensembling import Ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier

import sys
sys.setrecursionlimit(1500)

#You want to create variables X formatted as (nTrials x nTimepoints x nElectrodes) and Y formatted as (nTrials x 1)
X = []
y = []

Chans = X.shape[2]
Length = X.shape[1]
nbClasses = len(np.unique(y))

#======Level 1 Models======================
'''
First, we'll make models with a variety of parameters. These will serve as level one models.
The predictions of these level one models will later be combined.
'''
model1 = RCNN.init(Chans, Length, nbClasses, nbRCL=6, nbFilters=32, earlystopping=True, patience=20, filtersize=3, epochs=200)
model2 = RCNN.init(Chans, Length, nbClasses, nbRCL=6, nbFilters=64, earlystopping=True, patience=20, filtersize=3, epochs=200)
model3 = RCNN.init(Chans, Length, nbClasses, nbRCL=6, nbFilters=128, earlystopping=True, patience=20, filtersize=3, epochs=200)
model4 = RCNN.init(Chans, Length, nbClasses, nbRCL=6, nbFilters=150, earlystopping=True, patience=20, filtersize=3, epochs=200)

model5 = RCNN.init(Chans, Length, nbClasses, nbRCL=8, nbFilters=32, earlystopping=True, patience=20, filtersize=3, epochs=200)
model6 = RCNN.init(Chans, Length, nbClasses, nbRCL=8, nbFilters=64, earlystopping=True, patience=20, filtersize=3, epochs=200)
model7 = RCNN.init(Chans, Length, nbClasses, nbRCL=8, nbFilters=128, earlystopping=True, patience=20, filtersize=3, epochs=200)
model8 = RCNN.init(Chans, Length, nbClasses, nbRCL=8, nbFilters=150, earlystopping=True, patience=10, filtersize=3)

model9 = RCNN.init(Chans, Length, nbClasses, nbRCL=7, nbFilters=33, earlystopping=True, patience=20, filtersize=3, epochs=200)
model10 = RCNN.init(Chans, Length, nbClasses, nbRCL=7, nbFilters=6, earlystopping=True, patience=20, filtersize=3, epochs=200)
model11 = RCNN.init(Chans, Length, nbClasses, nbRCL=7, nbFilters=128, earlystopping=True, patience=20, filtersize=3, epochs=200)
model12 = RCNN.init(Chans, Length, nbClasses, nbRCL=7, nbFilters=150, earlystopping=True, patience=20, filtersize=3, epochs=200)

model_list = [model1, model2, model3, model4,
              model5, model6, model7, model8,    
              model9, model10, model11, model12,]
                 
#==== Second Level: Blending Models ====================
rf = RandomForestClassifier(n_estimators = 200)
svm0 = svm.SVC(decision_function_shape='ovo', probability=True)
trees = ExtraTreesClassifier(max_depth=3, n_estimators=200, random_state=0)
sgd = SGDClassifier(loss="modified_huber", penalty="l2")
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
booster = AdaBoostClassifier(n_estimators=200)

meta_classifiers = [rf, svm0, trees, sgd, bagging, booster]
#======================

X_train, y_train, X_test, y_test = utils.train_test_splitter(X, Y)

## Model Blending
'''
Here we demonstrate how to do soft voting (averaging), hard voting, and model blending. First we create our model ensemble
with 'Ensemble()'. We next fit our model. If you want to do model blending, you'll have to also fit 
the meta level models with  'fit_meta()'. For voting ensembles, we use the call 'majority_vote'. To blend models we call
'blend()'
'''
sclf = Ensemble(classifiers=model_list, meta_classifier = meta_classifiers, use_probas=True, verbose=1)
sclf.fit(X_train, y_train, X_test, y_test)
sclf.fit_meta(X_train, y_train)

Predictions_soft = sclf.majority_vote(X_test, voting='soft')
Predictions_hard = sclf.majority_vote(X_test, voting='hard')
Predictions_blended = sclf.blend(X_test)

accuracy_soft= accuracy_score(y_test, Predictions_soft)  
accuracy_hard = accuracy_score(y_test, Predictions_hard)
accuracy_blend = accuracy_score(y_test, Predictions_blended)  

print("\nAccuracy: (Blended)%s  (Soft)%s  (Hard)%s \n" %(accuracy_blend, accuracy_soft, accuracy_hard))    

