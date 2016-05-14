'''
Some utility functions.
Very unorganized currently...
'''


import numpy as np
import sys
import matlab.engine
import os
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import Normalizer
from sklearn.externals import six
from sklearn.cross_validation import train_test_split
from importlib import import_module
from nolearn.lasagne import NeuralNet, PrintLayerInfo
from lasagne.updates import adam, nesterov_momentum, adagrad, adadelta, adamax
from nolearn.lasagne import BatchIterator
#from unbalanced_dataset import OverSampler

def MakeEEGData(params):
    #Run Matlab scripts to process and return data (These will be subject & experiment specific)
    #This is presently very specific to a project - will need to be made a generic function
    #You want to create variables X formatted as (nTrials x nTimepoints x nElectrodes) and Y formatted as (nTrials x 1)
    
    eng = matlab.engine.start_matlab()
    Output = eng.GetMouseEEG(params['samplerate'],  params['eegfile'], async=True, nargout=3)
    Result = Output.result()
    
    EEG = np.array(Result[0][::])
    EMG = np.array(Result[1][::])
    Y = np.array(Result[2][::])
    del Result
    
    if params['normalize']==True:       
        normalizer = Normalizer(copy=False)
        EEG = normalizer.fit_transform(EEG)
        EMG = normalizer.fit_transform(EMG)
    
    if params['channels']==1:
        X = EEG[...,np.newaxis] 
    if params['channels']==2:
        X = np.concatenate((EEG[...,np.newaxis], EMG[...,np.newaxis]), axis=2)

    ix = np.where(np.logical_and(Y>=1, Y<=3))
    Y = Y[ix[0]]
    X = X[ix[0],:,:]
    
    X = np.float32(X)
    Y = np.int32(Y)

    Y-=1 #Set to zero index since it's coming from matlab
     
        
    return X, Y
    
def MakeSegments(X,Y, params, mode = 'train'):
        
    #Make an array with the indexes of each class type
    ClassStats = np.unique(Y,return_counts=True)
    ClassLocs=[] 
    for i in range(ClassStats[0].shape[0]): 
        ClassLocs.append( [ind for ind, exam in enumerate(X) if Y[ind] == ClassStats[0][i]] )
        
    SegmentStartArray = range(0, X.shape[1]-params['SegmentLength'])
    
    #Loop by class and allocate random segments of initial signals
    X_Segments=[]; Y_Segments=[]
    for i in range(ClassStats[0].shape[0]):
        
        #If test mode, scale number of segments based on input class
        if mode == 'train':
            nSegments = params['nbSegments']
        elif mode == 'test':
            nSegments = ClassStats[1][i]*2
        
        X_j= np.zeros((nSegments, params['SegmentLength'], X.shape[2]))
        Y_j = np.zeros(nSegments)     
        for j in range(nSegments):
            SegmentStartNum = np.random.choice(SegmentStartArray)
            SegmentIdx = np.random.choice(ClassLocs[i])
            X_j[j,:,:] = X[SegmentIdx, SegmentStartNum:SegmentStartNum+params['SegmentLength'], :] 
            Y_j[j] = Y[SegmentIdx,:]
            
        X_Segments.append(X_j)
        Y_Segments.append(Y_j)
        
    X_new = np.concatenate(X_Segments[:], axis=0)
    Y_new = np.concatenate(Y_Segments[:], axis=0)
    
    return X_new, Y_new


def SplitData(X,Y, train=None, test=None):

    if train is not None:
        X_train_fold = X[train]
        X_test_fold = X[test]    
        Y_train_fold = Y[train]
        Y_test_fold = Y[test]
    else:
        X_train_fold, X_test_fold, Y_train_fold, Y_test_fold = train_test_split(X, Y, test_size = params['testsize'])
    
    # if params['bootstrap']==True:
    #     X_train_fold, Y_train_fold = Oversample(X_train_fold, Y_train_fold)
    
    # if params['MakeSegments']==True:
    #     X_train_fold, Y_train_fold  = MakeSegments(X_train_fold, Y_train_fold, params, mode = 'train') 
    #     X_test_fold, Y_test_fold = MakeSegments(X_test_fold, Y_test_fold, params, mode = 'test') 
        
                        
    X_train_fold = X_train_fold[...,np.newaxis]
    X_test_fold = X_test_fold[...,np.newaxis]
    
    X_train_fold = np.float32(np.transpose(X_train_fold, [0,2,3,1]))
    X_test_fold = np.float32(np.transpose(X_test_fold, [0,2,3,1]))

    Y_train_fold =np.squeeze(np.int32(Y_train_fold))  
    Y_test_fold = np.squeeze(np.int32(Y_test_fold))

    return X_train_fold, Y_train_fold, X_test_fold, Y_test_fold

def Oversample(X, y):
        '''
        Oversample underrepresented class
        '''
                
        ClassStats = np.unique(y,return_counts=True)
        
        MinClass = np.argmin(np.unique(y,return_counts=True)[1])
        MaxClass = np.argmax(np.unique(y,return_counts=True)[1])
        AllOtherClasses = ClassStats[0][np.logical_and(ClassStats[0]!=MinClass,ClassStats[0]!=MaxClass)]
        nsamples = np.round(np.mean(ClassStats[1][AllOtherClasses])) - ClassStats[1][MinClass]
        if nsamples > ClassStats[1][MinClass]: nsamples = np.round(ClassStats[1][MinClass]*.9)

        ClassLocs=[] 
        for i in range(ClassStats[0].shape[0]): 
            ClassLocs.append( [ind for ind, exam in enumerate(X) if y[ind] == ClassStats[0][i]] )
          
        oversample_idx = np.random.choice(ClassLocs[MinClass] ,size = nsamples, replace=True)

        X_oversample = X[oversample_idx, :, :]   
        Y_oversample = y[oversample_idx]  

        X_new = np.concatenate((X, X_oversample), axis=0)   
        Y_new = np.concatenate((y, Y_oversample), axis=0) 
        
        return X_new, Y_new
       

def MakeModels(model_list, X, params):
    '''
    Make models and put in a list so that we can use them for a voting classifier
    '''  
    compiled_models = []  
    for i, model in enumerate(model_list):
        model0 = import_module('models.%s' % model)
        reload(model0)
        net = model0.build_model(X,params['nb_classes'])
        
        net0 = NeuralNet(
            net['l_out'],
            max_epochs=params['max_epochs'],
             
            #update=nesterov_momentum,
            #update_learning_rate=0.01,
            #update_momentum=0.9,
            update=adam,
            #update_learning_rate=0.0002,
            objective_l2=0.0025,
            
            on_epoch_finished=[EarlyStopping(patience=10)],

            batch_iterator_train = BatchIterator(batch_size=64),
            batch_iterator_test = BatchIterator(batch_size=64),    
            verbose=params['verbose'],
        )
        clf = (str(model), net0)
        #clf = (net0)
        
        compiled_models.append(clf)
        
    return compiled_models

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):

    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        

class VotingClassifier(NeuralNet, ClassifierMixin, TransformerMixin):
    """
    Voting classfier adapted from scikit-learn for neural network input from nolearn models
    example usage:
    
    compiled_models = utils.MakeModels(model_list, X, params)
    eclf = utils.VotingClassifier(estimators=compiled_models)
    eclf = eclf.fit(X, y, X_val, y_val)
    Predictions = eclf.predict(X_val, voting='hard')
    
    """

    def __init__(self, estimators, weights=None):

        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.weights = weights

    def fit(self, X, y, X_val, y_val):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')
#
#        if self.voting not in ('soft', 'hard'):
#            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
#                             % self.voting)

        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        #self.le_ = LabelEncoder()
        #self.le_.fit(y)
        #self.classes_ = self.le_.classes_
        self.estimators_ = []

        for name, clf in self.estimators:
            print('Training model %s...'%(name)),            
            fitted_clf = clf.fit(X, y, X_val, y_val)
            fitted_clf_score = fitted_clf.score(X_val, y_val)
            print(fitted_clf_score)

            self.estimators_.append(fitted_clf)

        return self

    def predict(self, X, voting='hard'):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        #maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        #if self.voting == 'hard':
        #    raise AttributeError("predict_proba is not available when"
        #                         " voting=%r" % self.voting)
        return self._predict_proba

    def transform(self, X, voting='hard'):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        if voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(VotingClassifier, self).get_params(deep=False)
        else:
            out = super(VotingClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T
        

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping."),
            #print("Best valid loss was {:.6f} at epoch {}.".format(
            #    self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()