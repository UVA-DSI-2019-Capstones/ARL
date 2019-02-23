# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:20:54 2019

@author: Erfaneh
"""
#%% imports
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as classification_report
from scipy import sparse
from nltk.corpus import wordnet
from nltk import word_tokenize
#from T_SNE import plotDocs
import string
import spacy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os

#%%
nlp = spacy.load('en')

#%% colors for t-sne plot

colors = ['green', 'gold', 'crimson', 'darkblue', 'orangered', 'orchid', 'olive', 'dodgerblue', 'yellowgreen', 'lime', 'salmon', 'darkseagreen', 'wheat']
classes = ["1.00", "1.33", "1.67", "2.00", "2.33", "2.67", "3.00", "3.33", "3.67", "4.00", "4.33", "4.67", "5.00"]
labelColor = {}
for i in range(0, len(classes)):
    labelColor[classes[i]] = colors[i]

#%% Functions
def Extractor(text, POS):
    nouns = []
    text2 = nlp(text)
    for word in text2:
        if(word.pos_ == POS):
            nouns.append(str(word))
    #print(len(nouns))
    return " ".join(nouns)

def FilterPunc(doc):
    exclude = set(string.punctuation)
    #stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in doc if ch not in exclude)
    #normalized = "".join(lemma.lemmatize(word) for word in punc_free.split())
    return punc_free

def syn_DicCreator (X_train_new, POS):
    syns_Dic = {}
    for text in X_train_new:
        words = Extractor(FilterPunc(text).lower(), POS).split(" ")
        for word in words:
            if word not in syns_Dic:
                syns = wordnet.synsets(word, pos = POS[0].lower())
                syns_Name = list(set([synset.lemmas()[0].name() for synset in syns]))
                if word in syns_Name:
                    syns_Name.remove(word) 
                if (len(syns_Name)!= 0):
                    syns_Dic[word] = syns_Name
    return syns_Dic

#%%
import os
import os.path
os.chdir('..')
print( os.path.basename(os.getcwd()) )
#%%
os.chdir('..')
os.chdir(os.path.join(dir, 'db'))
#%% Read Data
df = pd.read_csv("C:\\Users\\Erfaneh\\Google Drive\\Projects\\Army\\Database\\response.csv", delimiter = '\t')
df = df.fillna(" ")

#%% 
X = df.turk_response_text
Y = df.response_score
    
test_response = df[['response_text', 'response_score']]
test_response = test_response.drop_duplicates()
X_test = test_response.response_text
Y_test = test_response.response_score

X = X.append(X_test)
Y = Y.append(Y_test)
#%% score clustering
Y = Y.replace(1.33, 1)
Y = Y.replace(1.67, 1)
Y = Y.replace(2, 1)
Y = Y.replace(2.33, 1)
Y = Y.replace(2.67, 3)
Y = Y.replace(3.33, 3)
Y = Y.replace(3.67, 5)
Y = Y.replace(4, 5)
Y = Y.replace(4.33, 5)
Y = Y.replace(4.67, 5)


    
#%% Seprate test and train data
    
X = X.tolist()
Y = Y.tolist()

X_train_new,X_test,Y_train_new,Y_test = train_test_split(X,Y,test_size=0.20)
Y_train_str = Y_train_new #['{:.2f}'.format(x) for x in Y_train_new]
Y_test_str = Y_test #['{:.2f}'.format(x) for x in Y_test]


#%% Dictionaries of Synonyms
synDic_Noun = syn_DicCreator(X_train_new, 'NOUN')
synDic_Verb = syn_DicCreator(X_train_new, 'VERB')
synDic_Adj = syn_DicCreator(X_train_new, 'ADJ')

#%% Data resampling
new_X = []
new_Y = []

for i in range(0, len(X_train_new)):
    text = FilterPunc(X_train_new[i]).lower().replace("\"", "")
    new_X.append(text)
    new_Y.append(Y_train_new[i])
    words = nlp(text)
    for word in words:
        if(word.pos_ == 'NOUN'): 
            word = str(word)
            if (word in synDic_Noun.keys()):
                for syn in synDic_Noun[word]:
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train_new[i])                    
            continue
        if(word.pos_ == 'VERB'):
            word = str(word)
            if word in synDic_Verb:
                for syn in synDic_Verb[word]:
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train_new[i])
            continue
        if(word.pos_ == 'ADJ'):
            word = str(word)
            if word in synDic_Verb:
                for syn in synDic_Verb[word]:
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train_new[i])  

X_train = new_X
Y_train_str = new_Y

#%% Classification

for dim in range (100, 2100, 100):

    print (dim)
    
    #Data vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features = dim)
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train).toarray()
    tfidf_matrix_Test = tfidf_vectorizer.transform(X_test).toarray()    
    
    #Plotting
    #plotDocs(tfidf_matrix, Y_train_str, "train", labelColor)
    #plotDocs(tfidf_matrix_Test, Y_test_str, "test", labelColor)

    #Classification Algorithms 
    from sklearn.naive_bayes import MultinomialNB
    print("NaiveBayes", end = ' ')
    clf = MultinomialNB().fit(tfidf_matrix, Y_train_str)
    predicted = clf.predict(tfidf_matrix_Test)
    print(classification_report(Y_test_str, predicted, average='weighted'))
    
    from sklearn.neural_network import MLPClassifier
    print("MLP", end = ' ')
    mlp = MLPClassifier(hidden_layer_sizes=(50),max_iter=500)
    #Y_train1 = np.asarray(Y_train, dtype=np.float64).tolist()
    mlp.fit(tfidf_matrix,Y_train_str)
    predicted = mlp.predict(tfidf_matrix_Test)
    print(classification_report(Y_test_str, predicted, average='weighted'))
    
    from sklearn import tree
    print("tree", end = ' ')
    clf = tree.DecisionTreeClassifier()
    clf.fit(tfidf_matrix,Y_train_str)
    predicted = clf.predict(tfidf_matrix_Test)
    print(classification_report(Y_test_str, predicted, average='weighted'))
    
    from sklearn.ensemble import RandomForestClassifier
    print("RandomForest", end = ' ')
    clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
    clf.fit(tfidf_matrix,Y_train_str)
    predicted = clf.predict(tfidf_matrix_Test)
    print(classification_report(Y_test_str, predicted, average='weighted'))
