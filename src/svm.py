import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from text_cleaner import *

# Train model
# Split the data
train, test = get_train_test('../data/kaggle/labeledTrainData.tsv')

# Clean training data
print 'Cleaning training data'
clean_train_reviews = clean_tweets(train['review'])

# Creating the TFIDF model
vectorizer = TfidfVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             ngram_range = (1, 3),
                             stop_words = 'english',
                             sublinear_tf = True,
                             max_features = 5000)

# Fit the Bag of Words model
# Create feature vectors for every review
print 'Fit bag of words and create feature vectors for training data'
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print 'Fit SVM with cross validation'
clf = svm.SVC(kernel='linear', C=1).fit(train_data_features, train['sentiment'])
scores = cross_val_score(clf, train_data_features, train['sentiment'], cv=5)

joblib.dump(clf, '../models/svm.pkl')

print scores

# Load models
# clf = joblib.load('../models/svm.pkl')
# vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Clean testing data
print 'Cleaning testing data'
clean_test_reviews = clean_tweets(test['review'])

# Create feature vectors for every review
print 'Create feature vectors for testing data'
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the svm to make sentiment label predictions
result = clf.predict(test_data_features)

cnt = 0
i = 0
for sent in test['sentiment']:
  i += 1
  if sent == result[i]:
    cnt += 1

print 'Total matches: {0}'.format(cnt)
