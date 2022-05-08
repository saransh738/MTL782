#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

#Reading data and converting into a dataframe
data_points = pd.read_csv("file.csv",nrows = 58510)
print(data_points.boxplot())
#Shuffling the data 
data_points = data_points.sample(frac=1)
data = np.array(data_points.values)
dp = np.array(data)


#features holds all the attributes information of all the class_labels
features = dp[:,:48]
scl_p1 = max(features.max(), -features.min())
features = features/scl_p1
#class_label holds classes
class_label = dp[:,48]


# In[5]:


#printing covariance
print(data_points.cov())


# In[6]:


#printing correlation
print(data_points.corr())


# In[ ]:




