#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_model import model 
# Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[2]:


batch_size = 1
num_classes = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


class TestDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(np.load('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_x.npy').astype(np.float32))
        self.y = torch.from_numpy(np.load('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_y.npy').reshape(-1,1).astype(np.float32))
        self.n_samples = self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

testDataset = TestDataset()
test_loader = DataLoader(dataset=testDataset,batch_size=batch_size,shuffle=True)


# In[4]:


print(f"Number of samples: {len(test_loader)}")


# In[5]:


model = model.to(device)
model.load_state_dict(torch.load('/Users/wajidabdul/courses/cis579/code/muze/torch/model.pt'))
print(model)
# Set model to evaluation model
model.eval() 


# In[6]:


# Dictionary genres
dict_genres = {
    "Hip-Hop": 0,
    "International": 1,
    "Electronic": 2,
    "Folk" : 3,
    "Experimental": 4,
    "Rock": 5,
    "Pop": 6,
    "Instrumental": 7  
}


# In[7]:


y_true = []
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs/255
        inputs = (inputs[:,None,:,:]).to(device)
        labels = F.one_hot(targets.type(torch.LongTensor),num_classes).reshape(-1,num_classes).to(device)
        out = model(inputs)
        _, predicted = torch.max(out.data, 1)
        y_true.append((torch.max(labels,1)[1]).tolist())
        y_pred.append(predicted.tolist())


# In[8]:


y_true[:10]


# In[9]:


y_pred[:10]


# In[10]:


from sklearn.metrics import classification_report
target_names = dict_genres.keys()
print(classification_report(y_true, y_pred, target_names=target_names))


# In[11]:





mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.2)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=dict_genres.keys(),
            yticklabels=dict_genres.keys())
plt.xlabel('True label', fontsize=20, fontweight='bold')
plt.ylabel('Predicted label', fontsize=20, fontweight='bold');
plt.show();


# In[12]:


from sklearn.metrics import accuracy_score
print("Accuracy score: {:.4f} % ".format(accuracy_score(y_true, y_pred)*100))


# In[ ]:




