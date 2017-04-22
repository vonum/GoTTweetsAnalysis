import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from text_cleaner import *

# Train model
# Load training data
train, test = get_train_test('../data/kaggle/labeledTrainData.tsv')
print train.shape
print test.shape

'''
# Clean data for training and testing
print 'Cleaning data for training'
clean_train_reviews = clean_tweets(train['review'])

# Creating the TFIDF model
vectorizer = TfidfVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             ngram_range = (1, 3),
                             max_features = 3000)

# Fit the Bag of Words model
# Create feature vectors for every review
print 'Fit bag of words and vectorize data'
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Load model
# forest = joblib.load('../models/cross_validated_random_forest.pkl')
# vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Fit the Random Forest classifier with training data features and sentiment labels
print 'Fit random forests model'
forest = forest.fit(train_data_features, train['sentiment'])

# Save models
joblib.dump(forest, '../models/cross_validated_random_forest.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
'''

forest = joblib.load('../models/cross_validated_random_forest.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

print 'Cleaning data for testing'
clean_test_reviews = clean_tweets(test['review'])

# Transform test reviews from sentences to feature vectors
# Convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
print 'Predict results'
result = forest.predict(test_data_features)

cnt = 0
for i in xrange(0, len(test['sentiment'])):
  if test['sentiment'].iloc[0] == result[i]:
    cnt += 1

print 'Total matches: {0}'.format(cnt)
