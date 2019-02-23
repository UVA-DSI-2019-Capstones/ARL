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
from scipy.spatial.distance import cosine 
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
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

def syn_DicCreator (X_train, POS):
    syns_Dic = {}
    for text in X_train:
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
dir = os.getcwd()

dim = 200

from pathlib import Path

#%%
#Get working directory
dir = Path(os.getcwd())
#Move up two directories
# p = Path(dir).parents[1]
print(dir)

#%%
#Open the word vectors:
WV = {}
wvpack = "glove.6B."+str(dim)+"d.txt"
file_1 = dir / "glove.6B" / wvpack
#%%
df = pd.read_csv(file_1, sep=" ", quoting=3, header=None, index_col=0)
print(df.head())
WV = {key: val.values for key, val in df.T.items()}


#%% Read Data
train_df = pd.read_csv(dir / 'train_set.csv', delimiter = ',')
test_df = pd.read_csv(dir / 'test_set.csv', delimiter = ',')
# df = df.fillna(" ")

#%% 
print(train_df.columns)
X_train = train_df['response_text']
Y_train = train_df['response_round_score']
X_test = test_df['response_text']
Y_test = test_df['response_round_score']

print(X_train.shape)
print(X_test.shape)


# #%% Dictionaries of Synonyms
# synDic_Noun = syn_DicCreator(X_train, 'NOUN')
# synDic_Verb = syn_DicCreator(X_train, 'VERB')
# synDic_Adj = syn_DicCreator(X_train, 'ADJ')

#%%
import numpy as np

# # Save
# np.save('synDic_Noun.npy', synDic_Noun) 
# np.save('synDic_Verb.npy', synDic_Verb)
# np.save('synDic_Adj.npy', synDic_Adj)

#%%
synDic_Noun = np.load('synDic_Noun.npy').item()
synDic_Verb = np.load('synDic_Verb.npy').item()
synDic_Adj = np.load('synDic_Adj.npy').item()

#%% Data resampling
new_X = []
new_Y = []
similarity_level = []

for i in range(0, len(X_train)):
    text = FilterPunc(X_train[i]).lower().replace("\"", "")
    new_X.append(text)
    new_Y.append(Y_train[i])
    words = nlp(text)
    for word in words:
        if(word.pos_ == 'NOUN'): 
            word = str(word)
            if (word in synDic_Noun.keys()):
                for syn in synDic_Noun[word]:
                    syn_token = nlp(syn)
                    word_token = nlp(word)
                    similarity_level.append(word_token.similarity(syn_token))
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train[i])                    
            continue
        if(word.pos_ == 'VERB'):
            word = str(word)
            if word in synDic_Verb:
                for syn in synDic_Verb[word]:
                    syn_token = nlp(syn)
                    word_token = nlp(word)
                    similarity_level.append(word_token.similarity(syn_token))
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train[i])
            continue
        if(word.pos_ == 'ADJ'):
            word = str(word)
            if word in synDic_Verb:
                for syn in synDic_Verb[word]:
                    syn_token = nlp(syn)
                    word_token = nlp(word)
                    similarity_level.append(word_token.similarity(syn_token))
                    new_X.append(text.replace(word, syn))
                    new_Y.append(Y_train[i])  

X_train_new = new_X
Y_train_new = new_Y

#%%
df = pd.DataFrame([X_train_new, Y_train_new, similarity_level]).transpose()
print(df.head())

#%%
df.to_csv('rephrasings.csv')

# #%% Classification

# for dim in range (100, 2100, 100):

#     print (dim)
    
#     #Data vectorization
#     tfidf_vectorizer = TfidfVectorizer(max_features = dim)
#     tfidf_matrix = tfidf_vectorizer.fit_transform(X_train).toarray()
#     tfidf_matrix_Test = tfidf_vectorizer.transform(X_test).toarray()    
    
#     #Plotting
#     #plotDocs(tfidf_matrix, Y_train_str, "train", labelColor)
#     #plotDocs(tfidf_matrix_Test, Y_test_str, "test", labelColor)

#     #Classification Algorithms 
#     from sklearn.naive_bayes import MultinomialNB
#     print("NaiveBayes", end = ' ')
#     clf = MultinomialNB().fit(tfidf_matrix, Y_train_str)
#     predicted = clf.predict(tfidf_matrix_Test)
#     print(classification_report(Y_test_str, predicted, average='weighted'))
    
#     from sklearn.neural_network import MLPClassifier
#     print("MLP", end = ' ')
#     mlp = MLPClassifier(hidden_layer_sizes=(50),max_iter=500)
#     #Y_train1 = np.asarray(Y_train, dtype=np.float64).tolist()
#     mlp.fit(tfidf_matrix,Y_train_str)
#     predicted = mlp.predict(tfidf_matrix_Test)
#     print(classification_report(Y_test_str, predicted, average='weighted'))
    
#     from sklearn import tree
#     print("tree", end = ' ')
#     clf = tree.DecisionTreeClassifier()
#     clf.fit(tfidf_matrix,Y_train_str)
#     predicted = clf.predict(tfidf_matrix_Test)
#     print(classification_report(Y_test_str, predicted, average='weighted'))
    
#     from sklearn.ensemble import RandomForestClassifier
#     print("RandomForest", end = ' ')
#     clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
#     clf.fit(tfidf_matrix,Y_train_str)
#     predicted = clf.predict(tfidf_matrix_Test)
#     print(classification_report(Y_test_str, predicted, average='weighted'))
