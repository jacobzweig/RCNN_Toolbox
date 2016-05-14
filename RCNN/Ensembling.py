# Ensembling methods for RCNN toolbox
# by Jacob Zweig - 2016
# Allows stacking and voting classifiers for Convolutional Neural Networks
#
# Modified from Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions


from nolearn.lasagne import NeuralNet
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.pipeline import _name_estimators
from sklearn.externals import six
import numpy as np
import sys


class Ensemble(NeuralNet, ClassifierMixin, TransformerMixin):
    
    """A Stacking and Voting classifier for scikit-learn estimators for classification.
    
    Example:
        
    sclf = Ensemble(classifiers=model_list, meta_classifier = meta_classifiers)
    sclf.fit(X, y)
    sclf.fit_meta(X, y)
    
    Predictions_soft = sclf.majority_vote(X_test, voting='soft')
    Predictions_hard = sclf.majority_vote(X_test, voting='hard')
    Predictions_blended = sclf.blend(X_test)
    
    
    Parameters
    ----------
    classifiers : array-like, shape = [n_regressors]
        A list of classifiers.
        Invoking the `fit` method on the `StackingClassifer` will fit clones
        of these original classifiers that will
        be stored in the class attribute
        `self.clfs_`.
    meta_classifier : list of classifers (1 to infinite, well, not infinite), optional
        The meta-classifier/s to be fitted on the ensemble of
        classifiers. If no meta classifiers are provided, we're assuming you're making a voting classifier
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
    Attributes
    ----------
    clfs_ : list, shape=[n_classifiers]
        Fitted classifiers (clones of the original classifiers)
    meta_clf_ : estimator
        Fitted meta-classifier (clone of the original meta-estimator)
    """
    def __init__(self, classifiers, meta_classifier=None,
                 use_probas=False, verbose=0):

        if meta_classifier is not None:        
            self.meta_classifier = meta_classifier
            self.named_meta_classifier = {'meta-%s' % key: value for
                                      key, value in _name_estimators([meta_classifier])}
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}


        self.use_probas = use_probas
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        """ Fit ensemble classifers and the meta-classifier.
        """
        self.clfs_ = []
        if self.verbose > 0:
            print("Fitting %d classifiers..." % (len(self.classifiers)))

        i=1
        for clf in self.classifiers:
            
            if self.verbose > 0:
                print("Fitting classifier %d of %d:" %
                      (i, len(self.classifiers))),

            if self.verbose > 2:
                if hasattr(clf, 'verbose'):
                    clf.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(clf)
            
            sys.stdout.flush() #flush output to make sure it's printing 
#            if X_val is not None:
#                fitted_clf = clf.fit(X, y, X_val, y_val)
#            else:
#                fitted_clf = clf.fit(X, y)
            fitted_clf = clf.fit(X, y)
            
            self.clfs_.append(fitted_clf)
            
#            if X_val is not None:
            fitted_clf_score = fitted_clf.score(X_val, y_val)
            print(fitted_clf_score)
                
            sys.stdout.flush() #flush output to make sure it's printing 
            i+=1
            
        return self
            
    def fit_meta(self, X, y):
        self.meta_clfs_ = []

        meta_features = self._predict_meta_features(X)
        for meta_clf in self.meta_classifier:
            fitted_meta_clf = meta_clf.fit(meta_features, y)
            self.meta_clfs_.append(fitted_meta_clf)

        return self


    def _predict_meta_features(self, X):
        if self.use_probas:
            probas = np.asarray([clf.predict_proba(X) for clf in self.clfs_])
            vals = np.average(probas, axis=0)
        else:
            vals = np.asarray(([clf.predict(X) for clf in self.clfs_])).T
        return vals

    def blend(self, X):
        """ Predict target values for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        labels : array-like, shape = [n_samples]
            Predicted class labels.
        """
        meta_features = self._predict_meta_features(X)
        predictions = np.asarray(([meta_clf.predict(meta_features) for meta_clf in self.meta_clfs_])).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=1, arr=predictions)

        return maj        

    def majority_vote(self, X, voting='hard'):
        self.voting = voting  
        probas_temp = self.use_probas
                
        if self.voting == 'soft':
            self.use_probas = True
            meta_features = self._predict_meta_features(X)
            maj = np.argmax(meta_features, axis=1)
            self.use_probas = probas_temp
            
        elif self.voting=='hard':
            self.use_probas=False
            meta_features = self._predict_meta_features(X)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=1, arr=meta_features)
            self.use_probas = probas_temp

        return maj
    
    def predict_proba(self, X):
        """ Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        proba : array-like, shape = [n_samples, n_classes]
            Probability for each class per sample.
        """
        meta_features = self._predict_meta_features(X)
        
        #probas = np.asarray([meta_clf.predict_proba(meta_features) for meta_clf in self.meta_clfs_])
        #vals = np.average(probas, axis=0)
        vals = np.concatenate(([meta_clf.predict_proba(meta_features) for meta_clf in self.meta_clfs_]), axis=1)

        return vals

