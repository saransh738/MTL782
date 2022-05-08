#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

###########################################################################

#Tuning Hyperparameters
#Hyperparameters are Max_Depth ,minimum number of samples required to be at a leaf node and The function to measure the quality of a split.

#using countour plots and varying Max_Depth and minimum number of samples required to be at a leaf node 
#keeping criterion as gini index

m1 = np.zeros((125,1))
m2 = np.zeros((125,1))
Max_Depth = np.linspace(1, 25, 25)
Min_Samples_Leaf = [1,2,3,4,5]

for i in range (25):
    for j in range(5):
        DTC = DecisionTreeClassifier(criterion = "gini",max_depth=Max_Depth[i], min_samples_leaf=Min_Samples_Leaf[j])
        DTC.fit(features, class_label)
        m1[5*i+j][0] = DTC.score(features, class_label)
        m2[5*i+j][0] = crossval_score(DTC, features,class_label, cv = 5)
       
        
Min_Samples_Leaf, Max_Depth = np.meshgrid(Min_Samples_Leaf, Max_Depth)
graph = np.ravel(m1)
matrix = np.ravel(m2)
matrix = matrix.reshape(Max_Depth.shape)
fig, p = plt.subplots()
k = p.contourf(Max_Depth, Min_Samples_Leaf, matrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Max_Depth and Min_Samples_Leaf (cross-validation)')
plt.xlabel('Max_Depth')
plt.ylabel('Min_Samples_Leaf')
plt.show()

graph = graph.reshape(Max_Depth.shape)
fig, p = plt.subplots()
k = p.contourf(Max_Depth, Min_Samples_Leaf, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Max_Depth and Min_Samples_Leaf (Training)')
plt.xlabel('Max_Depth')
plt.ylabel('Min_Samples_Leaf')
plt.show()


# In[4]:


#using countour plots and varying Max_Depth and minimum number of samples required to be at a leaf node 
#keeping criterion as entropy index

E1 = np.zeros((125,1))
E2 = np.zeros((125,1))
Max_Depth = np.linspace(1, 25, 25)
Min_Samples_Leaf = [1,2,3,4,5]

for i in range (25):
    for j in range(5):
        DTC = DecisionTreeClassifier(criterion = "entropy",max_depth=Max_Depth[i], min_samples_leaf=Min_Samples_Leaf[j])
        DTC.fit(features, class_label)
        E1[5*i+j][0] = DTC.score(features, class_label)
        E2[5*i+j][0] = crossval_score(DTC, features,class_label, cv = 5)
       
        
Min_Samples_Leaf, Max_Depth = np.meshgrid(Min_Samples_Leaf, Max_Depth)
graph1 = np.ravel(E1)
matrix1 = np.ravel(E2)
matrix1 = matrix1.reshape(Max_Depth.shape)
fig, p = plt.subplots()
k = p.contourf(Max_Depth, Min_Samples_Leaf, matrix1)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Max_Depth and Min_Samples_Leaf (cross-validation)')
plt.xlabel('Max_Depth')
plt.ylabel('Min_Samples_Leaf')
plt.show()

graph1 = graph1.reshape(Max_Depth.shape)
fig, p = plt.subplots()
k = p.contourf(Max_Depth, Min_Samples_Leaf, graph1)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Max_Depth and Min_Samples_Leaf (Training)')
plt.xlabel('Max_Depth')
plt.ylabel('Min_Samples_Leaf')
plt.show()


# In[14]:


#Printing accuracy for the best hyperparameters

DTC1 = DecisionTreeClassifier(criterion = "gini",max_depth=20, min_samples_leaf=1)
DTC1.fit(features,class_label)
print("Training Score @Gini_Index: ",DTC1.score(features, class_label))
print("Cross_Validation Score @Gini_Index: ",crossval_score(DTC1,features,class_label,5))

DTC2 = DecisionTreeClassifier(criterion = "entropy",max_depth=18, min_samples_leaf=1)
DTC2.fit(features,class_label)
print("Training Score @Entropy: ",DTC2.score(features, class_label))
print("Cross_Validation Score @Entropy: ",crossval_score(DTC2,features,class_label,5))


# In[ ]:




