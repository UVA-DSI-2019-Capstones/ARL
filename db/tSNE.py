
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wva = np.load("data\\Question1test_50dimensions.npy")
labels = np.load("data\\test_labels_question1.npy")

#%%
X_train = scale(wva, axis = 1)


#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2).fit_transform(X_train)
#%%

plt.scatter(tsne[:, 0], tsne[:, 1], s = 50, label = labels)


#%%
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('T-SNE Plot Glove Average 50 Dimensions', fontsize = 20)
label_names = np.unique(labels)
for label in label_names:
    idx = labels == label
    plt.scatter(tsne[idx, 0], tsne[idx, 1], s = 50, label = label)
plt.legend()

