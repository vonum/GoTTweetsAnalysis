import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

from text_cleaner import review_to_words

'''
# Train model
# Load training data
train = pd.read_csv('../data/kaggle/labeledTrainData.tsv',
                    header=0,
                    delimiter='\t',
                    quoting=3)

print train.columns.values

num_reviews = train['review'].size

# Save transformed reviews in list
clean_train_reviews = []

for i in xrange(0, num_reviews):
  if((i+1) % 1000 == 0):
    print 'Review %d of %d\n' % (i+1, num_reviews)

  clean_train_reviews.append(review_to_words(train['review'][i]))

print 'Cleaned review'
print clean_train_reviews[0]

# Creating the Bag of Words model
vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
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

print train_data_features.shape

# Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the Random Forest classifier with training data features and sentiment labels
forest = forest.fit(train_data_features, train['sentiment'])
# Save model
joblib.dump(forest, 'random_forest.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
'''

# Load model
forest = joblib.load('random_forest.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test the model
# Read the test data
df = pd.read_csv('../data/kaggle/labeledTrainData.tsv',
                   header=0,
                   delimiter='\t',
                   quoting=3)

train, test = train_test_split(df, test_size = 0.2)

print test.columns.values

# Create an empty list and append the clean reviews one by one
num_reviews = len(test['review'])
clean_test_reviews = []

print 'Cleaning and parsing the test set movie reviews...\n'
i = 0
for review in test['review']:
  if((i+1) % 1000 == 0):
    print 'Review %d of %d\n' % (i+1, num_reviews)
  i += 1

  clean_test_reviews.append(review_to_words(review))

# Transform test reviews from sentences to feature vectors
# Convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
