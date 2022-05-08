#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
data_points= data_points.sample(frac=1)
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
#Hyperparameters are Max_Depth ,Number of Trees in the forest and The function to measure the quality of a split.

#using countour plots and varying Max_Depth and Number of Trees in the forest
#keeping criterion as gini index

m1 = np.zeros((375,1))
m2 = np.zeros((375,1))
Estimators = np.linspace(1, 25, num=25,dtype=int)
Max_depth = np.linspace(6, 20, num=15,dtype=int)

for i in range (25):
    for j in range(15):
        RF = RandomForestClassifier(n_estimators = Estimators[i],criterion='gini',max_depth=Max_depth[j])
        RF.fit(features, class_label)
        m1[15*i+j][0] = RF.score(features, class_label)
        m2[15*i+j][0] = crossval_score(RF, features,class_label, cv = 5)
       
        
Max_depth, Estimators  = np.meshgrid(Max_depth, Estimators )
graph = np.ravel(m1)
matrix = np.ravel(m2)
matrix = matrix.reshape(Estimators .shape)
fig, p = plt.subplots()
k = p.contourf(Estimators,Max_depth, matrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Estimators and Max_depth (cross-validation)')
plt.xlabel('Estimators')
plt.ylabel('Max_depth')
plt.show()

graph = graph.reshape(Estimators .shape)
fig, p = plt.subplots()
k = p.contourf(Estimators, Max_depth, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Estimators and Max_depth (Training)')
plt.xlabel('Estimators')
plt.ylabel('Max_depth')
plt.show()


# In[4]:


#using countour plots and varying Max_Depth and Number of Trees in the forest
#keeping criterion as entropy

E1 = np.zeros((375,1))
E2 = np.zeros((375,1))
Estimators = np.linspace(1, 25, num=25,dtype=int)
Max_depth = np.linspace(6, 20, num=15,dtype=int)

for i in range (25):
    for j in range(15):
        RF = RandomForestClassifier(n_estimators = Estimators[i],criterion='entropy',max_depth=Max_depth[j])
        RF.fit(features, class_label)
        E1[15*i+j][0] = RF.score(features, class_label)
        E2[15*i+j][0] = crossval_score(RF, features,class_label, cv = 5)
       
        
Max_depth, Estimators  = np.meshgrid(Max_depth, Estimators )
graph1 = np.ravel(E1)
matrix1 = np.ravel(E2)
matrix1 = matrix1.reshape(Estimators.shape)
fig, p = plt.subplots()
k = p.contourf(Estimators,Max_depth, matrix1)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Estimators and Max_depth (cross-validation)')
plt.xlabel('Estimators')
plt.ylabel('Max_depth')
plt.show()

graph1 = graph1.reshape(Estimators.shape)
fig, p = plt.subplots()
k = p.contourf(Estimators, Max_depth, graph1)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s Estimators and Max_depth (Training)')
plt.xlabel('Estimators')
plt.ylabel('Max_depth')
plt.show()


# In[22]:


#Printing accuracy for the best hyperparameters

RF1 = RandomForestClassifier(n_estimators =25 ,criterion='gini',max_depth=20)
RF1.fit(features,class_label)
print("Training Score @Gini_Index: ",RF1.score(features, class_label))
print("Cross_Validation Score @Gini_Index: ",crossval_score(RF1,features,class_label,5))

RF2 = RandomForestClassifier(n_estimators =25 ,criterion='entropy',max_depth=20)
RF2.fit(features,class_label)
print("Training Score @Entropy: ",RF2.score(features, class_label))
print("Cross_Validation Score @Entropy: ",crossval_score(RF2,features,class_label,5))


# In[ ]:




