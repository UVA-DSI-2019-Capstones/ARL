import pandas as pd
import os
os.chdir('db')
from pathlib import Path
import sys
#Package to pre process
import gensim
from gensim.utils import simple_preprocess
from gensim.models import ldamodel
from gensim.test.utils import datapath
import numpy as np
from gensim.models import Word2Vec
from shorttext.utils import standard_text_preprocessor_1
pre = standard_text_preprocessor_1()
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')

# df_test = pd.DataFrame('test_set.csv')

def get_process_data_frame(data_frame):
  # Preprocessing
  data_frame['processed'] = data_frame['response_text'].apply(pre)
  corpus = data_frame['processed'].apply(lambda x: x.split(' '))
  return corpus


train_corpus = get_process_data_frame(train)
test_corpus = get_process_data_frame(test)

def get_word_2_vec_df(corpus, dim = 100):
  # train word2vec model
  # Default dimensions is 100
  # sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).
  model = Word2Vec(corpus, min_count=1)
  # summarize the loaded model
  # print(model)

  # summarize vocabulary
  # words = list(model.wv.vocab)
  # print(words)

  # access vector for one word
  # print(model['think'])

  mean_corpus = pd.DataFrame(corpus.apply(lambda x: np.mean(model[x], axis=0)))

  return pd.DataFrame(mean_corpus['processed'].values.tolist(), index = mean_corpus.index)

dir = os.getcwd()
for dim in range(100, 501, 50):
  train_set = get_word_2_vec_df(train_corpus, dim)
  test_set = get_word_2_vec_df(test_corpus, dim)

  train_set = pd.concat([train_set, train['response_round_score']], axis=1)
  test_set = pd.concat([test_set, test['response_round_score']], axis=1)

  train_file = "word_2_vec_train{}.csv".format(dim)
  test_file = "word_2_vec_test{}.csv".format(dim)

  train_set.to_csv(os.path.join(dir, 'word2vecmodels', train_file))
  test_set.to_csv(os.path.join(dir, 'word2vecmodels', test_file))

for dim in range(10, 91, 20):
  train_set = get_word_2_vec_df(train_corpus, dim)
  test_set = get_word_2_vec_df(test_corpus, dim)

  train_set = pd.concat([train_set, train['response_round_score']], axis=1)
  test_set = pd.concat([test_set, test['response_round_score']], axis=1)

  train_file = "word_2_vec_train{}.csv".format(dim)
  test_file = "word_2_vec_test{}.csv".format(dim)

  train_set.to_csv(os.path.join(dir, 'word2vecmodels', train_file))
  test_set.to_csv(os.path.join(dir, 'word2vecmodels', test_file))


