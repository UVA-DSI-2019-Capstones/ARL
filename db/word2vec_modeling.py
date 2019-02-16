import pandas as pd
import os
import glob
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
os.chdir(os.path.join(os.getcwd(),'db','word2vecmodels'))

train_files = [i for i in glob.glob('*train*.{}'.format('csv'))]
test_files = [i for i in glob.glob('*test*.{}'.format('csv'))]

train_test_list = list(zip(train_files, test_files))

for train, test in train_test_list:
  train_df = pd.read_csv(train)
  test_df = pd.read_csv(test)

  x_train = train_df.iloc[:,1:-1]
  y_train = train_df['response_round_score']

  x_test = test_df.iloc[:, 1:-1]
  y_test = test_df['response_round_score']

  clf = LogisticRegressionCV(cv=10, random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train,
                                                                                                   y_train)
  score = clf.score(x_test, y_test)
  print(str(''.join([n for n in train if n.isdigit()][1:]) + ' ' + str(score) + ' ' +
            str(f1_score(y_test, clf.predict(x_test), average = 'weighted'))))