import pandas as pd

from bs4 import BeautifulSoup
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def review_to_words(review):
  # 1. Remove HTML
  review_text = BeautifulSoup(review, 'lxml').get_text()

  # 2. Remove non-letters
  letters_only = re.sub('[^a-zA-Z]', ' ', review_text)

  # 3. Convert to lower case, split into individual words
  words = letters_only.lower().split()

  # 4. In Python, searching a set is much faster than searching
  #    a list, so convert the stop words to a set
  stops = set(stopwords.words('english'))

  lemmatizer = WordNetLemmatizer()
  # 5. Remove stop words
  # Lemmatize words
  meaningful_words = [lemmatizer.lemmatize(w) for w in words if not w in stops]

  # 6. Join the words back into one string separated by space,
  # and return the result.
  return( ' '.join( meaningful_words ))
