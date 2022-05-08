#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
    
#function for cross_validation
#it takes model,complete training data and cross folds as input   
def crossval_score(model, X, Y, cv = 5):

    n = X.shape[0]
    accuracy = np.array([])
    # bs is the batch size
    bs = n//cv
    
    for i in range(cv):
    
        begin, end = i*bs, (i+1)*bs
        
        testing_X, testing_Y = X[begin:end,:], Y[begin:end]
        training_X, training_Y = np.delete(X, range(begin, end), axis = 0), np.delete(Y, range(begin, end))
        
        model.fit(training_X, training_Y)
        prediction = model.predict(testing_X)
        
        f1 = f1_score(testing_Y, prediction , average='weighted')
        
        accuracy = np.append(accuracy, f1)
        
    return  accuracy.mean()

#Reading data and converting into a dataframe
data_points = pd.read_csv("file.csv",nrows = 58510)
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

####################################################################

#Tuning Hyperparameters
#Hyperparameter is var_smoothing i.e, Portion of the largest variance of all features 
# that is added to variances for calculation stability

fig = plt.figure(1)
matrix1 = np.zeros((35,3))
for i in range (1,35):
    
    Naiyes = GaussianNB(priors=None, var_smoothing=np.exp(-(i)))
    Naiyes.fit(features, class_label)
    matrix1[i][0] = i
    matrix1[i][1] = crossval_score(Naiyes, features,class_label, cv = 5)
    matrix1[i][2] = Naiyes.score(features, class_label)
    
plt.plot(matrix1[:,0:1],matrix1[:,1:2],label = 'cross_validation score')
plt.plot(matrix1[:,0:1],matrix1[:,2:3],label = 'Training score')
plt.title('Var_Smoothing vs Accuracy')
plt.xlabel('Var_Smoothing')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


#Printing accuracy for the best hyperparameters

Naiyes = GaussianNB(priors=None, var_smoothing=np.exp(-21))
Naiyes.fit(features,class_label)
print("Training Score: ",Naiyes.score(features, class_label))
print("Cross_Validation Score : ",crossval_score(Naiyes,features,class_label,5))


# In[ ]:




