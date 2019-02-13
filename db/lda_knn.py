from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
train = pd.read_csv('train_set.csv')
train['processed'] = train['response_text'].apply(pre)
train_corpus = train['processed'].apply(lambda x : x.split(' '))
dict_train = Dictionary(train_corpus)
bow_corpus_train = [dict_train.doc2bow(doc) for doc in train_corpus]

df_results = pd.DataFrame(columns = ['number_of_topics', 'n_neigh', 'accuracy', 'f1'])

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
        
    print('Start')
    
    test_df = test_df.transpose()
    train_df = train_df.transpose()
    
    for n_neigh in range(3,number_of_topics):
        neigh = KNeighborsClassifier(n_neighbors=n_neigh)
        neigh.fit(test_df, test['response_round_score']) 
        
        y_pred = neigh.predict(test_df)
        print('Number of topics {}'.format(number_of_topics))
        f1 = f1_score(test['response_round_score'], y_pred, average = 'weighted')
        accuracy = accuracy_score(test['response_round_score'], y_pred)
            
        df_results = pd.concat([df_results, pd.DataFrame([[number_of_topics, n_neigh, accuracy, f1]], columns = ['number_of_topics', 'n_neigh', 'accuracy', 'f1'])], axis = 0)

df_results.to_csv('k_nn_results.csv')
