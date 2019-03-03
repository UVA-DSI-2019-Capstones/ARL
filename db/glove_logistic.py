#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from gensim.corpora.dictionary import Dictionary
from pathlib import Path
import matplotlib.pyplot as plt
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

#%%
dim = [50, 100, 200, 300]
#%% Set path to Glove word vector folder
dir = Path(os.getcwd())

for i in range(len(dim)):
    wvpack = "glove.6B."+str(dim[i])+"d.txt"
    file_1 = dir / "glove.6B" / wvpack

    df = pd.read_csv(file_1, sep=" ", quoting=3, header=None, index_col=0)
    WV = {key: val.values for key, val in df.T.items()}
    file_2 = 'wv_dic_{}.npy'.format(dim[i])
    np.save(file_2, WV) 



#%%
#Create test set corpus
test = pd.read_csv('test_set.csv')
train = pd.read_csv('train_set.csv')
table = str.maketrans({key: None for key in string.punctuation})
X_test = test['response_text'].apply(lambda x : x.lower().translate(table))
X_train = train['response_text'].apply(lambda x : x.lower().translate(table))
Y_train = train['response_round_score']
Y_test = test['response_round_score']




#%%
# #If we only want to use the responses for the first 16 questions 
# #(these are the only responses that have more than one label)
# full_corpus['identifier'] = full_corpus['identifier'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
# print(full_corpus.head())
# full_corpus = full_corpus[full_corpus['identifier'] < 17]


#%%
# def string_indices(df1, df2, x_df, str_list):
#     for i in range(len(str_list)):
#         error_indices = np.union1d(df1.index[df1 == str_list[i]], df2.index[df2 == str_list[i]])
#         x_df.drop(error_indices, inplace=True)
#         df1.drop(error_indices, inplace=True)
#         df2.drop(error_indices, inplace = True)
#         x_df.reset_index(drop=True, inplace = True)
#         df1.reset_index(drop=True, inplace = True)
#         df2.reset_index(drop=True, inplace = True)
#     return df1, df2, x_df


# #%% Creating test set from context data
# context_data = pd.read_csv('context_data.csv')

# for i in range(1,10)
#     df = context_data[context_data['Scection'] == i]
#     Y_test_me = context_data['Me']
#     Y_test_vb = context_data['Vaibhav']
#     X_test = context_data['Unlabeled Text']
#     X_test = X_test.apply(lambda x : x.lower().translate(table))



# print(context_data.index[context_data['Me'] == 'e'])
# print(context_data.index[context_data['Vaibhav'] == 'e'])
# Y_test_me, Y_test_vb, X_test = string_indices(Y_test_me, Y_test_vb, X_test, ['e', '?'])


# print(len(X_test))
# print(len(Y_test_me))
# print(len(Y_test_vb))
# # Y_test_me.drop(error_indices, inplace=True)
# # Y_test_vb.drop(error_indices, inplace = True)
# # Y_test_me.reset_index(drop=True, inplace = True)
# # Y_test_vb.reset_index(drop=True, inplace = True)
# print(Y_test_me.index[Y_test_me == 'e'])
# print(Y_test_me.index[Y_test_vb == 'e'])

# #%%
# unlabeled_values = np.intersect1d(np.where((pd.isna(Y_test_me))), np.where(pd.isna(Y_test_vb)))
# vb_label = np.intersect1d(np.where(pd.isna(Y_test_me)), np.where(~pd.isna(Y_test_vb)))

# #%%
# print(unlabeled_values)
# print(vb_label)


# #%%
# Y_test_me.drop(unlabeled_values, inplace=True)
# X_test.drop(unlabeled_values, inplace = True)
# Y_test_me.iloc[vb_label] = Y_test_vb.iloc[vb_label]
# print(len(Y_test_me))

# #%%
# print(len(X_test))
# #%%

#%%
for i in range(len(dim)):
    file_2 = 'wv_dic_{}.npy'.format(dim[i])
    WV = np.load(file_2).item() 
    trainingMatrix = np.zeros((0, dim[i]))
    testMatrix = np.zeros((0, dim[i]))

    for train_doc in X_train:
        trainingMatrix = np.append(trainingMatrix, [np.asarray(docAveraging(train_doc, WV, dim[i]))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)

    for test_doc in X_test:
        testMatrix = np.append(testMatrix, [np.asarray(docAveraging(test_doc, WV, dim[i]))], axis=0)#.decode('utf8').strip()), WV, dim))], axis=0)
    file_3 = 'trainingMatrix{}.npy'.format(dim[i])
    file_4 = 'testMatrix{}.npy'.format(dim[i])
    np.save(file_3, trainingMatrix) 
    np.save(file_4, testMatrix)

#%%
np.load('trainingMatrix.npy', trainingMatrix) 

#%%
X_train = scale(trainingMatrix, axis = 1)
X_test = scale(testMatrix, axis = 1)
# scaler = StandardScaler()
# scaler.fit(trainingMatrix)
# X_train = scaler.transform(trainingMatrix)
# scaler = StandardScaler()
# scaler.fit(testMatrix)
# X_test = scaler.transform(testMatrix)

#%%
print(Y_train)
Y_train_str = Y_train.to_string()
Y_test_str = Y_test.to_string()


#%%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_train)
#%%
X_embedded = pd.DataFrame(X_embedded, columns=['x', 'y'])

#%%
Y_train = pd.DataFrame(Y_train)
X_embedded = pd.concat([X_embedded, Y_train], axis=1)
#%%
colors = ['gold','darkseagreen', 'wheat']
# X_embedded = X_embedded.astype({'x': float, 'y': float, 'response_round_score': int})
classes = [1, 3, 5]
# labelColor = {}
# for i in range(0, len(classes)):
#     labelColor[classes[i]] = colors[i]

# plotDocs(X_embedded, Y_train_str, "train", labelColor)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('T-SNE Plot Glove Average 200 Dimensions', fontsize = 20)



for target, color in zip(classes,colors):
   indicesToKeep = X_embedded['response_round_score'] == target
   ax.scatter(X_embedded.loc[indicesToKeep, 'x'], X_embedded.loc[indicesToKeep, 'y'], c = color, s = 50, alpha=0.5)

ax.legend(classes)

ax.grid()
# plotDocs(tfidf_matrix_Test, Y_test_str, "test", labelColor)

#%%
import seaborn as sns
sns.lmplot(x = 'x', y = 'y', data = X_embedded)


#%%
clf = LogisticRegressionCV(cv =10,max_iter=500, random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, Y_train)
clf.predict(X_test)
score = clf.score(X_test, Y_test)
print('The score is {}'.format(score))
#%%
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train) 
y_pred = neigh.predict(X_test)
print(y_pred)

#%%
df_results = pd.DataFrame(columns = ['n_estimaters', 'depth', 'accuracy', 'f1', 'glove dimension'])
print('Start of Tree')
for i in range(len(dim)):
    # file_2 = 'wv_dic_{}.npy'.format(dim[i])
    # WV = np.load(file_2).item() 
    file_3 = 'trainingMatrix{}.npy'.format(dim[i])
    file_4 = 'testMatrix{}.npy'.format(dim[i])
    trainingMatrix = np.load(file_3) 
    testMatrix = np.load(file_4)
    for m_depth in range(2, 30, 2):
        print(m_depth)
        for n_est in range(100,300,50):
            print(str(m_depth) + ' - ' + str(n_est))
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=3214)
            clf.fit(trainingMatrix, Y_train)   
            y_pred = clf.predict(testMatrix)
            f1 = f1_score(Y_test, y_pred, average = 'weighted')
            accuracy = accuracy_score(Y_test, y_pred)
            print('Score: {}'.format(accuracy_score(Y_test, y_pred)))
            new_row = pd.DataFrame([n_est, m_depth, accuracy, f1, dim[i]])
            df_results = pd.concat([df_results, new_row.transpose()], axis = 0)


#%%
df_results.to_csv('glove_random_forests results.csv')