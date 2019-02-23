#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from gensim.corpora.dictionary import Dictionary
from pathlib import Path
import string
import pandas as pd
import numpy as np
import os
import re

#%%
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding = 'utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. " + str(len(model)) + " words loaded!")
    return model

#%%
def docAveraging(sent, WV, dim):
    summ = [0.0] * (dim)
    A = 0.0;
    sent_A = (re.sub(r"[\n(\[\])]", "", sent)).split(" ")
    for word in sent_A:
        if word in WV : #and word not in stop:
            A = A + 1.0
            for i in range(0, dim):
                summ[i] = summ[i] + float((WV[word])[i])
    if A != 0:
        #A = 1
        for i in range(0, dim):
            summ[i] = summ[i] / A
    return summ;

#%% Set path to Glove word vector folder
dir = Path(os.getcwd())
dim = 200
wvpack = "glove.6B."+str(dim)+"d.txt"
file_1 = dir / "glove.6B" / wvpack

df = pd.read_csv(file_1, sep=" ", quoting=3, header=None, index_col=0)
WV = {key: val.values for key, val in df.T.items()}

#%%
np.save('wv_dic_200.npy', WV) 

#%%
WV = np.load('wv_dic_200.npy').item()

#%%
#Create test set corpus
test = pd.read_csv('test_set.csv')
train = pd.read_csv('train_set.csv')
table = str.maketrans({key: None for key in string.punctuation})
test['response_text'] = test['response_text'].apply(lambda x : x.lower().translate(table))
train['response_text'] = train['response_text'].apply(lambda x : x.lower().translate(table))

full_corpus = pd.concat([test, train], axis = 0 )

#%%
#If we only want to use the responses for the first 16 questions 
#(these are the only responses that have more than one label)
full_corpus['identifier'] = full_corpus['identifier'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
print(full_corpus.head())
full_corpus = full_corpus[full_corpus['identifier'] < 17]

X_train = full_corpus['response_text']
Y_train = full_corpus['response_round_score']


#%%
def string_indices(df1, df2, x_df, str_list):
    for i in range(len(str_list)):
        error_indices = np.union1d(df1.index[df1 == str_list[i]], df2.index[df2 == str_list[i]])
        x_df.drop(error_indices, inplace=True)
        df1.drop(error_indices, inplace=True)
        df2.drop(error_indices, inplace = True)
        x_df.reset_index(drop=True, inplace = True)
        df1.reset_index(drop=True, inplace = True)
        df2.reset_index(drop=True, inplace = True)
    return df1, df2, x_df


#%% Creating test set from context data
context_data = pd.read_csv('context_data.csv')

for i in range(1,10):
    df = context_data[context_data['Scection'] == i]
    Y_test_me = context_data['Me']
    Y_test_vb = context_data['Vaibhav']
    X_test = context_data['Unlabeled Text']
    X_test = X_test.apply(lambda x : x.lower().translate(table))



print(context_data.index[context_data['Me'] == 'e'])
print(context_data.index[context_data['Vaibhav'] == 'e'])
Y_test_me, Y_test_vb, X_test = string_indices(Y_test_me, Y_test_vb, X_test, ['e', '?'])


print(len(X_test))
print(len(Y_test_me))
print(len(Y_test_vb))
# Y_test_me.drop(error_indices, inplace=True)
# Y_test_vb.drop(error_indices, inplace = True)
# Y_test_me.reset_index(drop=True, inplace = True)
# Y_test_vb.reset_index(drop=True, inplace = True)
print(Y_test_me.index[Y_test_me == 'e'])
print(Y_test_me.index[Y_test_vb == 'e'])

#%%
unlabeled_values = np.intersect1d(np.where((pd.isna(Y_test_me))), np.where(pd.isna(Y_test_vb)))
vb_label = np.intersect1d(np.where(pd.isna(Y_test_me)), np.where(~pd.isna(Y_test_vb)))

#%%
print(unlabeled_values)
print(vb_label)


#%%
Y_test_me.drop(unlabeled_values, inplace=True)
X_test.drop(unlabeled_values, inplace = True)
Y_test_me.iloc[vb_label] = Y_test_vb.iloc[vb_label]
print(len(Y_test_me))

#%%
print(len(X_test))
#%%

#%%
trainingMatrix = np.zeros((0, dim))
testMatrix = np.zeros((0, dim))

for train_doc in X_train:
    trainingMatrix = np.append(trainingMatrix, [np.asarray(docAveraging(train_doc, WV, dim))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)

for test_doc in X_test:
    testMatrix = np.append(testMatrix, [np.asarray(docAveraging(test_doc, WV, dim))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)


#%%
np.save('trainingMatrix.npy', trainingMatrix) 
np.save('testMatrix.npy', testMatrix)


#%%


#%%
for neighb in range(3,12):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(trainingMatrix, Y_train) 
    y_pred = neigh.predict(testMatrix)

    y_pred_2 = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            y_pred_2.append(3)
        elif y_pred[i] ==3:
            y_pred_2.append(2)
        elif y_pred[i] == 5:
            y_pred_2.append(1)
    Y_test = Y_test_me.astype(int).tolist()
    accuracy = accuracy_score(Y_test, y_pred_2)

    print('Accuracy for {} neighbors: {}'.format(neighb, accuracy))

#%%
print(len(y_pred))
#%%


#%%
print(len(y_pred_2))

#%%
Y_test = Y_test_me.astype(int).tolist()
#%%

# f1 = f1_score(test['response_round_score'], y_pred, average = 'weighted')
accuracy = accuracy_score(Y_test, y_pred_2)

#%%
print(accuracy)
#%%


# #Create training set corpus
# train = pd.read_csv('train_set.csv')
# train['processed'] = train['response_text'].apply(pre)
# train_corpus = train['processed'].apply(lambda x : x.split(' '))
# dict_train = Dictionary(train_corpus)
# bow_corpus_train = [dict_train.doc2bow(doc) for doc in train_corpus]

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
