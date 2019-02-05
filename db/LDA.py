import shorttext
import sqlite3
import pandas as pd
import os.path

conn = sqlite3.connect("database.db")
# df = pd.read_sql_query("select * from trainee_response;", conn)
df = pd.read_sql_query("SELECT * FROM trainee_response INNER JOIN turk_response ON trainee_response.identifier = turk_response.identifier;", conn)

# def score_round(score):
#     if score < 2.3:
#         return 1
#     elif score >= 2.3 and score < 4:
#         return 3
#     return 5

# df['response_round_score'] = df['response_score'].apply(score_round)


train = df.sample(frac=1, random_state=200)
test = df.drop(train.index)

#Preprocessing

from shorttext.utils import standard_text_preprocessor_1
pre = standard_text_preprocessor_1()

train['processed'] = train['response_text'].apply(pre)

corpus = train['processed'].apply(lambda x : x.split(' '))
index = list(corpus.index)

#Package to pre process
import gensim
from gensim.utils import simple_preprocess
from gensim.models import ldamodel
import numpy

#Builds a corpus of words
dictionary = gensim.corpora.Dictionary(corpus)
bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

numpy.random.seed(1)
upper_bound = 30
for i in range(2,upper_bound):
	if not(os.path.isfile('LDA_{}_topics.csv'.format(i))):
		lower_bound = i
		break

for number_of_topics in range(lower_bound,upper_bound):
	model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=number_of_topics)

	column_list = ["Original Input"]
	for i in range(number_of_topics):
		column_list.append("Topic {}".format(i+1))


	lda_df = pd.DataFrame(columns=column_list)


	for i, item in enumerate(bow_corpus):
		text = [str(df.loc[index[i], "response_text"])]
		doc_type, percent = map(list, zip(*model.get_document_topics(item, minimum_probability=0.000001)))
		print(model.get_document_topics(item, minimum_probability=0.000001))
		row = text + percent
		print(row)
		lda_df.loc[i] = row

		# for j in range(len(doc_type)):
		# 	print("document {} is {} type {}".format(i, percent[j], doc_type[j]))
		# print('\n')


	lda_df.to_csv('LDA_{}_topics.csv'.format(number_of_topics))
	print('Loop for {} topics complete'.format(number_of_topics))
	lda_df.drop(df.index, inplace=True)


    

