#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
features = dp[:,:26]
scl_p1 = max(features.max(), -features.min())
features = features/scl_p1
#class_label holds classes 
class_label = dp[:,48]


##########################################################################
#Tuning Hyperparameters
#Hyperparameters are Number of Neighbors and distance Metric

m1 = np.zeros((20,1))
m2 = np.zeros((20,1))
Neighbors = np.linspace(1, 10, num=10,dtype=int)
#We have taken assumption that if Metric value is 0 then distance metric chosen is Euclidean
#and if it is 1 then distance metric chosen is Manhattan
Metric = [0,1]

for i in range (10):
    for j in range(2):
        if(j==0):
            c = 'euclidean'
        else:
            c = 'manhattan'
        knn = KNeighborsClassifier(n_neighbors =Neighbors[i] ,metric = c)
        knn.fit(features, class_label)
        m1[2*i+j][0] = knn.score(features, class_label)
        m2[2*i+j][0] = crossval_score(knn, features,class_label, cv = 5)
        
Metric, Neighbors = np.meshgrid(Metric, Neighbors)
graph = np.ravel(m1)
matrix = np.ravel(m2)
matrix = matrix.reshape(Neighbors.shape)
fig, p = plt.subplots()
k = p.contourf(Neighbors, Metric, matrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Metric and Neighbors (cross-validation)')
plt.xlabel('Neighbors')
plt.ylabel('Metric')
plt.show()

graph = graph.reshape(Neighbors.shape)
fig, p = plt.subplots()
k = p.contourf(Neighbors, Metric, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Metric and Neighbors (Training)')
plt.xlabel('Neighbors')
plt.ylabel('Metric')
plt.show()


# In[4]:


#Printing accuracy for the best hyperparameters

knn = KNeighborsClassifier(n_neighbors = 3,metric='manhattan')
knn.fit(features,class_label)
print("Training Score: ",knn.score(features, class_label))
print("Cross_Validation Score: ",crossval_score(knn,features,class_label,5))


# In[ ]:




