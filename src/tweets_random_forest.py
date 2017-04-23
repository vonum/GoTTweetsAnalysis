import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from text_cleaner import *

# Train model
# Load tweets, clean them, and store in array
clean_train_tweets = []
train_sentiments = []

clean_test_tweets = []
test_sentiments = []

df = pd.read_csv('../data/random/amazon_cells_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

clean_train_tweets += clean_tweets(df['tweet'])
train_sentiments += format_sentiments(df['sentiment'])

df = pd.read_csv('../data/random/imdb_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

clean_train_tweets += clean_tweets(df['tweet'])
train_sentiments += format_sentiments(df['sentiment'])

df = pd.read_csv('../data/random/yelp_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

clean_train_tweets += clean_tweets(df['tweet'])
train_sentiments += format_sentiments(df['sentiment'])

df = pd.read_csv('../data/random/train_data.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

clean_train_tweets += clean_tweets(df['tweet'])
train_sentiments += format_sentiments(df['sentiment'])

print len(clean_train_tweets)

vectorizer = TfidfVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             ngram_range = (1, 3),
                             stop_words = 'english',
                             max_features = 5000)

# Fit the Bag of Words model
# Create feature vectors for every review
print 'Fit bag of words and create feature vectors for training data'
train_data_features = vectorizer.fit_transform(clean_train_tweets)
train_data_features = train_data_features.toarray()

train_sentiments = np.array(train_sentiments)

# test_data_features = vectorizer.fit_transform(clean_test_tweets)
# test_data_features = test_data_features.toarray()

print 'Fit SVM model'
# clf = svm.SVC(kernel='linear', C=1).fit(train_data_features, train_sentiments)
# scores = cross_val_score(clf, train_data_features, train_sentiments, cv=2)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 150,
                                random_state = 0,
                                max_features = 70)

# Fit the Random Forest classifier with training data features and sentiment labels
print 'Fit random forests model'
forest = forest.fit(train_data_features, train_sentiments)
# scores = cross_val_score(forest, train_data_features, train_sentiments)

# print scores

# Split data for training and testing 80% - 20%
sss = StratifiedShuffleSplit(n_splits = 3,
                             test_size = 0.2,
                             random_state = 0)
sss.get_n_splits(train_data_features, train_sentiments)

print 'Cross validation with 3 splits'
for train_index, test_index in sss.split(train_data_features, train_sentiments):
  X_train, X_test = train_data_features[train_index], train_data_features[test_index]
  y_train, y_test = train_sentiments[train_index], train_sentiments[test_index]
  # SVM
  # clf = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
  forest = RandomForestClassifier(n_estimators = 100,
                                random_state = 0,
                                max_features = 70).fit(X_train, y_train)
  print(forest.score(X_test, y_test))

'''
result = forest.predict(test_data_features)

cnt = 0
i = 0
for sent in test_sentiments:
  if sent == result[i]:
    cnt += 1
  i += 1

print cnt
'''
