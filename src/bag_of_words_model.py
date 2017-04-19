import pandas as pd

from bs4 import BeautifulSoup
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def review_to_words(review):
  # 1. Remove HTML
  review_text = BeautifulSoup(review).get_text()

  # 2. Remove non-letters
  letters_only = re.sub("[^a-zA-Z]", " ", review_text)

  # 3. Convert to lower case, split into individual words
  words = letters_only.lower().split()

  # 4. In Python, searching a set is much faster than searching
  #    a list, so convert the stop words to a set
  stops = set(stopwords.words("english"))

  lemmatizer = WordNetLemmatizer()
  # 5. Remove stop words
  # Lemmatize words
  meaningful_words = [lemmatizer.lemmatize(w) for w in words if not w in stops]

  # 6. Join the words back into one string separated by space,
  # and return the result.
  return( " ".join( meaningful_words ))

# Load training data
train = pd.read_csv("../data/kaggle/labeledTrainData.tsv",
                    header=0,
                    delimiter="\t",
                    quoting=3)

print train.columns.values

num_reviews = train['review'].size

# Save transformed reviews in list
clean_train_reviews = []

for i in xrange(0, num_reviews):
  if((i+1) % 1000 == 0):
    print "Review %d of %d\n" % (i+1, num_reviews)
  clean_train_reviews.append(review_to_words(train["review"][i]))

print 'Cleaned review'
print clean_train_reviews[0]

# Creating the Bag of Words model
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

# Fit the Bag of Words model
# Create feature vectors for every review
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print train_data_features.shape

# Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the Random Forest classifier with training data features and sentiment labels
forest = forest.fit(train_data_features, train["sentiment"])

# Test the model
# Read the test data
test = pd.read_csv("../data/kaggle/testData.tsv",
                   header=0,
                   delimiter="\t",
                   quoting=3)

print test.columns.values

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0, num_reviews):
  if( (i+1) % 1000 == 0 ):
    print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# Transform test reviews from sentences to feature vectors
# Convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
