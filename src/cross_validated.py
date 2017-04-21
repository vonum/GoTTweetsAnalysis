import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from text_cleaner import review_to_words

# Train model
# Load training data
df = pd.read_csv('../data/kaggle/labeledTrainData.tsv',
                    header=0,
                    delimiter='\t',
                    quoting=3)
# Split the data
train, test = train_test_split(df, test_size = 0.2)

clean_train_reviews = []

# Clean data
i = 0
for review in train['review']:
  if((i+1) % 1000 == 0):
    print 'Review %d\n' % (i+1)
  i += 1

  clean_train_reviews.append(review_to_words(review))

# Creating the TFIDF model
vectorizer = TfidfVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             ngram_range = (1, 6),
                             stop_words = 'english',
                             sublinear_tf = True,
                             max_features = 5000)

# Fit the Bag of Words model
# Create feature vectors for every review
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Load model
# forest = joblib.load('cross_validated_random_forest.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# Fit the Random Forest classifier with training data features and sentiment labels
forest = forest.fit(train_data_features, train['sentiment'])

joblib.dump(forest, 'cross_validated_random_forest.pkl')

clean_test_reviews = []

print 'Cleaning and parsing the test set movie reviews...\n'
i = 0
for review in test['review']:
  if((i+1) % 1000 == 0):
    print 'Review %d\n' % (i+1)
  i += 1

  clean_test_reviews.append(review_to_words(review))

# Transform test reviews from sentences to feature vectors
# Convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

cnt = 0
i = 0
for sent in test['sentiment']:
  if sent == result[i]:
    cnt += 1

print 'Total matches: {0}'.format(cnt)
