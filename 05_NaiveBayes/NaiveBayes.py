# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:59:00 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as pyplot

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.2)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#Navie Bayes model setup
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


classifier.fit(X_train, y_train)

#predict attempt 
NBpredict = classifier.predict(ss.transform([[29,80000]]))


y_predict = classifier.predict(X_test)


#Confusion Matrix and Accuracy Score
from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test, y_predict)
aScore = accuracy_score(y_test,y_predict)


