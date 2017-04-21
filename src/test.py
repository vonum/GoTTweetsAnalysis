import pandas as pd
import pdb

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/kaggle/labeledTrainData.tsv',
                    header=0,
                    delimiter='\t',
                    quoting=3)

train, test = train_test_split(df, test_size = 0.3)

for i in test['review']:
  print len(i)

print train.shape
print test.shape

print test['review'].shape
print test['id'].shape
print test['sentiment'].shape
