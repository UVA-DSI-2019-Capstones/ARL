from sklearn.ensemble import RandomForestClassifier
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

df_results = pd.DataFrame(columns = ['n_estimaters', 'depth', 'accuracy', 'f1','num_topics'])

#Load LDA model
for number_of_topics in range(2,30):
    temp_file = 'LDA_{}_topic.model'.format(number_of_topics)
    temp_file = os.path.join(dir, 'LDA_models', temp_file)
    lda = LdaModel.load(temp_file)
    
    
    
    
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    
    for i in range(0, len(bow_corpus_test)):
        test_df = pd.concat([test_df, pd.DataFrame(lda.get_document_topics(bow = bow_corpus_test[i], minimum_probability=0.000001))[1]], axis = 1)
        
    
    for i in range(0, len(bow_corpus_train)):
        train_df = pd.concat([train_df, pd.DataFrame(lda.get_document_topics(bow = bow_corpus_train[i], minimum_probability=0.000001))[1]], axis = 1)   
        
    
    
    test_df = test_df.transpose()
    train_df = train_df.transpose()
    print('Start of Tree')
    
    for m_depth in range(2, 30, 2):
        print(m_depth)
        for n_est in range(100,300,50):
            print(str(m_depth) + ' - ' + str(n_est))
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=3214)
            clf.fit(train_df, train['response_round_score'])
            
            y_pred = clf.predict(test_df)
            print('Number of topics {}'.format(number_of_topics))
            f1 = f1_score(test['response_round_score'], y_pred, average = 'weighted')
            accuracy = accuracy_score(test['response_round_score'], y_pred)
            print('Score: {}'.format(accuracy_score(test['response_round_score'], y_pred)))
            df_results = pd.concat([df_results, pd.DataFrame([n_est, m_depth, accuracy, f1, number_of_topics])], axis = 1)

df_results = df_results.transpose()
df_results.to_csv('Decision_trees_results.csv')
