# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:49:22 2025

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



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)


#ConfusionMatrix and AccuracyScore
from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test,y_pred)
aScore= accuracy_score(y_test,y_pred)