from sklearn.linear_model import LogisticRegressionCV
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from shorttext.utils import standard_text_preprocessor_1
import pandas as pd
import os

dir = os.getcwd()



#Create test set corpus
test = pd.read_csv('test_set.csv')
pre = standard_text_preprocessor_1()

test['processed'] = test['response_text'].apply(pre)
test_corpus = test['processed'].apply(lambda x : x.split(' '))

dict_test = Dictionary(test_corpus)
bow_corpus_test = [dict_test.doc2bow(doc) for doc in test_corpus]


#Create training set corpus
train = pd.read_csv('C:\\Users\\vaibhav\\Documents\\UVA\\Fall\\Capstone\\Code\\ARL\\db\\train_set.csv')
train['processed'] = train['response_text'].apply(pre)
train_corpus = train['processed'].apply(lambda x : x.split(' '))
dict_train = Dictionary(train_corpus)
bow_corpus_train = [dict_train.doc2bow(doc) for doc in train_corpus]


#Load LDA model
for number_of_topics in range(2,30):
    temp_file = 'LDA_{}_topic.model'.format(number_of_topics)
    temp_file = os.path.join(dir, 'LDA_models', temp_file)
    lda = LdaModel.load(temp_file)
    
    
    
    
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    
    print(bow_corpus_test[1])
    for i in range(0, len(bow_corpus_test)):
        test_df = pd.concat([test_df, pd.DataFrame(lda.get_document_topics(bow = bow_corpus_test[i], minimum_probability=0.000001))[1]], axis = 1)
        
    
    for i in range(0, len(bow_corpus_train)):
        train_df = pd.concat([train_df, pd.DataFrame(lda.get_document_topics(bow = bow_corpus_train[i], minimum_probability=0.000001))[1]], axis = 1)   
        
    
    
    test_df = test_df.transpose()
    train_df = train_df.transpose()
    
    clf = LogisticRegressionCV(cv =10, random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_df, train['response_round_score'])
    
    clf.predict(test_df)
    score = clf.score(test_df, test['response_round_score'])
    print('For {} topics, score is {}'.format(number_of_topics, score))