# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:54:02 2018

@author: Erfaneh
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.models import Model
# from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
# from keras.optimizers import RMSprop
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy import sparse
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

stop = []

dim = 200

import os
from pathlib import Path

#Get working directory
dir = Path(os.getcwd())
#Move up two directories
p = Path(dir).parents[1]


#Open the word vectors:
WV = {}
wvpack = "glove.6B."+str(dim)+"d.txt"
file_1 = p / "glove.6B" / wvpack


df = pd.read_csv(file_1, sep=" ", quoting=3, header=None, index_col=0)
print(df.head())
WV = {key: val.values for key, val in df.T.items()}


# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     f = open(gloveFile,'r', encoding = 'utf-8')
#     model = {}
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     print("Done. " + str(len(model)) + " words loaded!")
#     return model

# model = loadGloveModel(file_1)
# print(model['Hello'])

# for line in file_1:
#     line = line.replace("\n", "")
#     wordV = line.split(" ")
#     key = wordV[0]
#     if key not in stop:
#         del wordV[0]
#         WV[key] = np.asarray(wordV,dtype=float)


#Finding the word vector representation for a sentence by averaging the vector for each word
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

df = pd.read_csv(p / "Database/response_complete.csv")
df = df.fillna(" ")
X = df.loc[:209, ['turk_response_text']]
Y = df.loc[:209, ['response_score']]
X = X.reset_index(drop = True)
Y = Y.reset_index(drop=True)


print(X.head())
print(Y.head())
#sns.countplot(df.response_score)
#plt.xlabel('Score')
#plt.title('Number msgs for each Score')

Y[Y < 2] = 1
Y[(Y >=2) & (Y < 4)] = 3
Y[Y >= 4] = 5
# for i in range (0,len(Y)):
#     if Y.loc[i] <= 2:
#         Y.loc[i] = 5
#     elif  Y.loc[i] > 2 and Y[i] < 4:
#         Y.loc[i] = 3
#     else:
#         Y.loc[i] = 1



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)

print(X_train.shape)
print(X_test.shape)
print(X_train.head())


# #
# Y_train_str = ['{:.2f}'.format(x) for x in Y_train]
# Y_test_str = ['{:.2f}'.format(x) for x in Y_test]


trainingMatrix = np.zeros((0, dim))
testMatrix = np.zeros((0, dim))

# for train_doc in X_train:
#     trainingMatrix = np.append(trainingMatrix, [np.asarray(docAveraging(train_doc, WV, dim))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)

# for test_doc in X_test:
#     testMatrix = np.append(testMatrix, [np.asarray(docAveraging(test_doc, WV, dim))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)


#Create tfidv matrices
tfidf_vectorizer = TfidfVectorizer(max_features = dim)
tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
tfidf_matrix_Test = tfidf_vectorizer.transform(X_test)

print(tfidf_matrix.shape)


# print(testMatrix)
# tfidf_matrix = sparse.csr_matrix(trainingMatrix)
# tfidf_matrix_Test = sparse.csr_matrix(testMatrix)

print(tfidf_matrix)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(tfidf_matrix, Y_train)
predicted = clf.predict(tfidf_matrix_Test)
print(classification_report(Y_test, predicted))

# from sklearn import svm
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# clf.fit(tfidf_matrix, Y_train_str)
# predicted = clf.predict(tfidf_matrix_Test)
# print(classification_report(Y_test_str, predicted))


# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(tfidf_matrix,Y_train_str)
# predicted = clf.predict(tfidf_matrix_Test)
# print(classification_report(Y_test_str, predicted))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100),max_iter=500)
#Y_train1 = np.asarray(Y_train, dtype=np.float64).tolist()
mlp.fit(tfidf_matrix,Y_train)
predicted = mlp.predict(tfidf_matrix_Test)
print(classification_report(Y_test, predicted))


# #from sklearn import tree
# #clf = tree.DecisionTreeRegressor()
# #clf.fit(tfidf_matrix,[float(i) for i in Y_train_str])
# #predicted = clf.predict(tfidf_matrix_Test)
# #print(classification_report([float(i) for i in Y_test_str], predicted))




