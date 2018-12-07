# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:26:25 2018

@author: Admin
"""
#In this project, I use the Pokemon Dataset in Kaggle. I focus on the attribute Legendary. 
#I try to build a model to predict whether a specific Pokemon is Legendary or not 
#with five methods of classification methods in machine learning.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pkm = pd.read_csv('Pokemon.csv')
pkm.head()

#Variable Selection To implement PCA, I delete the 
#categorical attributes(type1 and type2) and other useless attributes(id, generation) first.
pkm1 = pkm.drop(['#','Name','Generation'],1)
pkm_attributes = pkm1.drop(['Legendary','Type 1','Type 2'],1)

from sklearn.decomposition import PCA
pca = PCA(n_components=pkm_attributes.shape[1])
fit = pca.fit(pkm_attributes).transform(pkm_attributes)

#Then I visualize the results and find the numbers of components used later.
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1 = np.insert(var1,0,0)
plt.plot(var1)
axes = plt.gca()
axes.set_ylim([40,110])
plt.show()

#Based on the changing point, I choose to set n=3.

pca = PCA(n_components=3)
fit = pca.fit(pkm_attributes).transform(pkm_attributes)
fit1 = pd.DataFrame(fit,columns=['c1','c2','c3'])
df = pd.concat([fit1,pkm1['Legendary']],axis=1)
df.head()

#df is the transformed attributes used in classification.
#Classification Model
#Split into train and test
#In the following part, I will make False = 1, True = 0 in Legendary.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

leg = abs((df['Legendary'].values - 1))
X_train, X_test, y_train, y_test = train_test_split(fit, leg, test_size=0.33, random_state=42)

#K-nearest Neighbor
#In this model, I try to find the best k for the number of neighborhoods.
from sklearn.neighbors import KNeighborsClassifier
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cross validation scores
cv_scores = []

# perform 10-fold cross validation we are already familiar with
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

#Then print the best k and visualize the MSE as before.

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

#Then build the best model of KNN and print the confusion matrix.

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train) 
y_predict_knn = knn.predict(X_test)
confusion_matrix(y_test, y_predict_knn)