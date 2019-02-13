import pandas as pd
import os
import pathlib
import sys


train = pd.read_csv('train_set.csv')
# df_test = pd.DataFrame('test_set.csv')


#Preprocessing

from shorttext.utils import standard_text_preprocessor_1
pre = standard_text_preprocessor_1()

train['processed'] = train['response_text'].apply(pre)

corpus = train['processed'].apply(lambda x : x.split(' '))
index = list(corpus.index)
print(corpus)

#Package to pre process
import gensim
from gensim.utils import simple_preprocess
from gensim.models import ldamodel
from gensim.test.utils import datapath
import numpy

#Builds a corpus of words
dictionary = gensim.corpora.Dictionary(corpus)
bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

dir = os.getcwd() 

upper_bound = 31
lower_bound = 2
for i in range(lower_bound,upper_bound):
	temp_file = "LDA_{}_topic.model".format(i)
	temp_file = os.path.join(dir,'LDA_models',temp_file)
	if not(os.path.isfile(temp_file)):
		# print(os.path.isfile('LDA_{}_topic.model'.format(i)))
		lower_bound = i
		break

if i == upper_bound-1:
	exit()



for number_of_topics in range(lower_bound,upper_bound):
	model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=number_of_topics)
	temp_file = "LDA_{}_topic.model".format(number_of_topics)
	temp_file = os.path.join(dir,'LDA_models',temp_file)
	model.save(datapath(temp_file))
	print('Loop for {} topics complete'.format(number_of_topics))