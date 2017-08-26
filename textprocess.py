import re
import string
import nltk.corpus
from sklearn.base import TransformerMixin


PUNCT = re.compile('[%s]' % re.escape(string.punctuation))
STOPS = set(nltk.corpus.stopwords.words('english'))


def strip_punct(content):
    ''' remove all punctuation
    '''
    return PUNCT.sub('', content)


def strip_numbers(content):
    ''' remove all digits
    '''
    return ''.join([i for i in content if not i.isdigit()])


def strip_stops(content):
    return ' '.join([x for x in content.split()
                     if x not in STOPS])


class KeySelect(TransformerMixin):
    ''' select key from dictionary
    '''
    def __init__(self, key='content'):
        self.key = key

    def transform(self, X, **transform_params):
        return [x[self.key] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self


class StripTransform(TransformerMixin):
    ''' transform raw content to only word chars
    '''
    def transform(self, X, y=None, **transform_params):
        return [strip_stops(strip_numbers(strip_punct(x)))
                 for x in X]

    def fit(self, X, y=None, **fit_params):
        return self


