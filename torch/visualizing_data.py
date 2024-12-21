#!/usr/bin/env python
# coding: utf-8

# In[17]:


import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[18]:


X = np.load("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_x.npy")
y = np.load("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_y.npy")





data = (X/255.).reshape(-1,128*128)
labels = y.reshape(-1,)
print(f"Data Shape: {data.shape}\nLabels Shape: {labels.shape}")




feat_cols = ['pixel'+str(i) for i in range(data.shape[1]) ]
df = pd.DataFrame(data,columns=feat_cols)
df['y'] = labels
df['label'] = df['y'].apply(lambda i: str(i))
df['label'].replace(to_replace=['0', '1', '2', '3', '4', '5', '6', '7'],
           value= ['Hip-Hop', 'International', 'Electronic', 'Folk', 'Experimental', 'Rock', 'Pop', 'Instrumental'], 
           inplace=True)
data, labels = None, None
print('Size of the dataframe: {}'.format(df.shape)); df




# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])




plt.gray()
fig = plt.figure( figsize=(16,10) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1)
    ax.set_title(r"Label: ${}$".format(str(df.loc[rndperm[i],'label'])))
    ax.imshow(df.loc[rndperm[i],feat_cols].values.reshape((128,128)).astype(float))
plt.show()


# In[23]:


N = 3998
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values ; len(data_subset)


# In[24]:


# Using t-SNE to reduce high-dimensional data "http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf"
time_start = time.time()
# configuring the parameteres
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=800 )
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[25]:


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_results.T, df_subset['label'])).T; tsne_data


# In[26]:


# Ploting the result of tsne
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sns.FacetGrid(tsne_df, hue="label").map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# In[ ]:




