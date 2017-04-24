import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from text_cleaner import clean_tweets

def filter_tweets(tweets, sentiment):
  pos_tweets = []

  for index, row in tweets.iterrows():
    if row['sentiment'] == sentiment:
      pos_tweets.append(row['tweet'])

  return pos_tweets

sentiment = int(sys.argv[1])

tweets = []

df = pd.read_csv('../data/random/amazon_cells_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

tweets += filter_tweets(df, sentiment)

df = pd.read_csv('../data/random/imdb_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

tweets += filter_tweets(df, sentiment)

df = pd.read_csv('../data/random/yelp_labelled.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

tweets += filter_tweets(df, sentiment)

df = pd.read_csv('../data/random/train_data.txt',
                 header=0,
                 delimiter='\t',
                 quoting=3)

tweets += filter_tweets(df, sentiment)

if sys.argv[2] == 'clean':
  tweets = clean_tweets(tweets)

text = ' '.join(tweets)

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
