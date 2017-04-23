import pandas as pd

from bs4 import BeautifulSoup
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split

def tweet_to_words(tweet):
  # 1. Remove HTML
  tweet_text = BeautifulSoup(tweet, 'lxml').get_text()

  # 2. Remove urls
  tweet_text = re.sub(r'http\S+', '', tweet_text)

  # 3. Change emojis with words
  tweet_text = replace_emojis(tweet_text)

  # 4. Remove @
  tweet_text = re.sub(r'@\S+', '', tweet_text)

  # 5. Remove #
  tweet_text = re.sub(r'#\S+', '', tweet_text)

  # 6. Remove non-letters
  letters_only = re.sub('[^a-zA-Z]', ' ', tweet_text)

  # 7. Convert to lower case, split into individual words
  words = letters_only.lower().split()

  # 8. In Python, searching a set is much faster than searching
  #    a list, so convert the stop words to a set
  stops = set(stopwords.words('english'))

  lemmatizer = WordNetLemmatizer()
  stemmer = SnowballStemmer('english')
  # 9. Remove stop words
  # Lemmatize words
  # meaningful_words = [lemmatizer.lemmatize(w) for w in words if not w in stops]
  meaningful_words = [stemmer.stem(w) for w in words if not w in stops]

  # 6. Join the words back into one string separated by space,
  # and return the result.
  return( ' '.join( meaningful_words ))

def get_train_test(file):
  df = pd.read_csv(file,
                   header=0,
                   delimiter='\t',
                   quoting=3)

  # Split the data
  train, test = train_test_split(df, test_size = 0.2)

  return train, test

def clean_tweets(tweets):
  clean_train_tweets = []

  # Clean each tweet
  i = 0
  for tweet in tweets:
    clean_train_tweets.append(tweet_to_words(tweet))

    i += 1
    if i % 1000 == 0:
      print 'Cleaned review {0}'.format(i)

  return clean_train_tweets

def replace_emojis(text):
 repls = {
   ':)' : 'happy',
   ':-)' : 'happy',
   ':]' : 'happy',
   ':-]' : 'happy',
   ':D' : 'laugh',
   ':-D' : 'laugh',
   ':=D' : 'laugh',
   'xD' : 'laugh',
   'XD' : 'laugh',
   ':(' : 'sad',
   ':-(' : 'sad',
   ':\'(' : 'sad',
   ':/' : 'sad',
   ':-/' : 'sad',
   ':o' : 'surprise',
   ':O' : 'surprise',
   '<3' : 'love',
   '</3' : 'hate',
   ':*' : 'kiss'
 }

 return reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), text)

def format_sentiments(sentiments):
  sents = []
  for sent in sentiments:
    sents.append(sent)

  return sents
