#!/usr/bin/env python
# coding: utf-8

# In[14]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
from sklearn.model_selection import train_test_split
import ast
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


# In[15]:


path = '/rscratch/xuanyu/KAIL/Kaggle_Doddle_Rank1/data/'#'/data/kaggle/doodle/'


# In[16]:


files = os.listdir(os.path.join(path, 'train_simplified'))


# In[17]:


data = []
for file in files:
    if '._' not in file:
        data.append(file)


# In[18]:


sample_num = 100000
valid_num = 80


# In[19]:


for file in data:
    cat = os.path.join(path, 'train_simplified/') + file
    df = pd.read_csv(cat, parse_dates=['timestamp'])
    #X_valid for valid
    X_train, X_valid, y_train, y_test = train_test_split(
        df, df['countrycode'], test_size=valid_num, random_state=1)
    #X_holdout for holdout
    X_keep, X_holdout, y_train, y_test = train_test_split(
        X_train, X_train['countrycode'], test_size=valid_num, random_state=2)
    #X_sample for train
    X_unuse, X_sample, y_train, y_test = train_test_split(
    X_keep, X_keep['countrycode'], test_size=sample_num, random_state=2)
    #outpur
    X_keep.to_csv(path + 'train_unuse/' + file, index=False)
    X_sample.to_csv(path + 'train_use/' + file, index=False)
    X_valid.to_csv(path + 'valid/' + file, index=False)
    X_holdout.to_csv(path + 'holdout/' + file, index=False)
    print(file, ' done!')


# In[ ]:





# In[ ]:




