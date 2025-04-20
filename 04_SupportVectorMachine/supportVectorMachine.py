# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:59:54 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#SVM model setup
from sklearn.svm import SVC
classifer = SVC(probability=True,kernel="rbf")


#Model learning
classifer.fit(X_train,y_train)


#Predict Attempt
SVMpredict=classifer.predict(ss.transform([[25,85000]]))



#Try on test set
y_pred = classifer.predict(X_test)
y_predPRobability = classifer.predict_proba(X_test)


#ConfusionMatrix and AccuracyScore
from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test,y_pred)
aScore= accuracy_score(y_test,y_pred)


