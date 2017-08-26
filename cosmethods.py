import re
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel


class CosineSim(BaseEstimator):
    """ store vector of target, predict by calculating cosine distance
    """ 
    def transform(self, X, **transform_params):
        return self

    def fit(self, X, y, **fit_params):
        """ X is a tfidf matrix of the training set
            reduce X to the tfidf of only documents in target class
        """ 
        indices = np.array([i for i in range(len(y)) if y[i] == 1])
        self.target_tfidf = X[indices, :].mean(axis=0)
        
    def predict(self, X, y=None, **predict_params):
        d = 1 - linear_kernel(X, self.target_tfidf).flatten()
        return d


class CosineSimTrans(TransformerMixin):
    """ store vector of target, predict by calculating cosine distance
    """    
    def transform(self, X, **transform_params):
        d = 1 - linear_kernel(self.target_tfidf, X).flatten()
        return d.reshape(-1, 1)

    def fit(self, X, y, **fit_params):
        indices = np.array([i for i in range(len(y)) if y[i] == 1])
        self.target_tfidf = X[indices, :].mean(axis=0)
        return self

