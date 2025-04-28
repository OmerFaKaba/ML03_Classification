# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:23:02 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#Model setup
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0,criterion="entropy")

#model learning
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)



#Confusion Matrix and Accucarcy Score
from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test, y_pred)
aScore = accuracy_score(y_test,y_pred)