import pandas as pd
import pdb

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/kaggle/labeledTrainData.tsv',
                    header=0,
                    delimiter='\t',
                    quoting=3)

train, test = train_test_split(df, test_size = 0.3)

tweets = []

df = pd.read_csv('../data/random/imdb_labelled.txt',
                header=0,
                delimiter='\t',
                quoting=3)

print df.shape
print df.columns.values

df = pd.read_csv('../data/random/yelp_labelled.txt',
                header=0,
                delimiter='\t',
                quoting=3)

print df.shape
print df.columns.values

df = pd.read_csv('../data/random/amazon_cells_labelled.txt',
                header=0,
                delimiter='\t',
                quoting=3)

print df.shape
print df.columns.values
