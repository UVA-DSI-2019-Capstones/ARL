import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
os.chdir(os.path.join(os.getcwd(),'db','word2vecmodels'))
import warnings
warnings.filterwarnings('always')

df_results = pd.DataFrame(columns=['n_estimaters', 'depth', 'accuracy', 'f1', 'dimensions'])


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
  print('Start of Tree')

  for m_depth in range(2, 30, 2):
    print(m_depth)
    for n_est in range(100, 300, 50):
      print(str(m_depth) + ' - ' + str(n_est))
      clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=3214)
      clf.fit(x_train, y_train)

      y_pred = clf.predict(x_test)
      f1 = f1_score(y_test, y_pred, average='weighted')
      accuracy = accuracy_score(y_test, y_pred)
      classifier_stats = pd.DataFrame([n_est, m_depth, accuracy, f1, str(''.join([n for n in train if n.isdigit()][1:]))]).transpose()
      classifier_stats.columns = ['n_estimaters', 'depth', 'accuracy', 'f1', 'dimensions']
      df_results = pd.concat([df_results, classifier_stats], axis=0)



print(df_results)
df_results.to_csv('random_forest_results.csv', ap)