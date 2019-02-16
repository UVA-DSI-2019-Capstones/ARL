import pandas as pd
import os
import glob
from sklearn.linear_model import LogisticRegressionCV
os.chdir('word2vecmodels')

train_files = [i for i in glob.glob('*train*.{}'.format('csv'))]
test_files = [i for i in glob.glob('*test*.{}'.format('csv'))]

train_test_list = list(zip(train_files, test_files))

for train_df, test_df in train_test_list:
  clf = LogisticRegressionCV(cv=10, random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_df, train[
    'response_round_score'])

  clf.predict(test_df)
  score = clf.score(test_df, test['response_round_score'])
  print('For {} topics, score is {}'.format(number_of_topics, score))