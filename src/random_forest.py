import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

from text_cleaner import *

# Train model
# Load training data
train, test = get_train_test('../data/kaggle/labeledTrainData.tsv')

print train.shape
print test.shape

# Clean train data
print 'Cleaning training data'
clean_train_reviews = clean_tweets(train['review'])

# Clean test data
print 'Cleaning test data'
clean_test_reviews = clean_tweets(test['review'])

print 'Cleaned review'
print clean_train_reviews[0]

# Creating the Bag of Words model
vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

# Fit the Bag of Words model
# Create feature vectors for every review
print 'Fit bag of words and create feature vectors'
vectorizer.fit(clean_train_reviews + clean_test_reviews)
train_data_features = vectorizer.transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the Random Forest classifier with training data features and sentiment labels
print 'Fit random forests model'
forest = forest.fit(train_data_features, train['sentiment'])

# Save model
joblib.dump(forest, '../models/random_forest.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')

# Load model
# forest = joblib.load('../models/random_forest.pkl')
# vectorizer = joblib.load('../models/vectorizer.pkl')

# Predict results for training data
result = forest.predict(test_data_features)

cnt = 0
i = 0
for sent in test['sentiment']:
  if sent == result[i]:
    cnt += 1
  i += 1

print 'Total matches: {0}'.format(cnt)
