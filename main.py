
"""
"""

import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion

from pdftojsonl import process_pdfs
from textprocess import KeySelect, StripTransform
from cosmethods import CosineSim, CosineSimTrans


def cossim(target, candidate):
    """ calculate cosine similarity between candidate and target
    """
    pipe = Pipeline([
        ('bykey', KeySelect()),
        ('clean', StripTransform()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('cosine', CosineSim())
    ])

    return 1 - pipe.fit(target, None).predict(candidate)


def cosclass(train_X, train_y, test_X):
    """ sgd classifier with cosine distance feature 
    """
    pipe = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('trans1', Pipeline([
                        ('tfidf', TfidfVectorizer())
                    ])),
                    ('trans2', Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('cosim', CosineSimTrans()),
                    ]))
                ]
            )),
            ('estimators', Pipeline([
                        ('clf', SGDClassifier(loss='perceptron', penalty='l2',
                                              alpha=1e-3, n_iter=5, random_state=42))
                        ])
            )
    ])

    return pipe.fit(train_X, train_y).predict(test_X)

