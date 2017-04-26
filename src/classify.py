import sys

import numpy as np
import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.externals import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from text_cleaner import *

# Load ensemble
vectorizer = joblib.load('../models/ensemble_vectorizer.pkl')
ensemble = joblib.load('../models/ensemble.pkl')

char = sys.argv[1]
df = pd.read_csv('../data/got/{0}.txt'.format(char),
                 header=0,
                 delimiter=';',
                 quoting=3)

print df.shape

clean_got_tweets = clean_tweets(df['text'])

tweets_features = vectorizer.transform(clean_got_tweets)
tweets_features = tweets_features.toarray()

results = ensemble.predict(tweets_features)

print Counter(results)
