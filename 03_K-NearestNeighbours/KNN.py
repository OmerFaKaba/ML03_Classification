# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:27:40 2025

@author: omer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1)

#feature scaling
from sklearn.preprocessing import StandardScaler

ss= StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



#KNN model
from sklearn.neighbors import KNeighborsClassifier

classifer = KNeighborsClassifier(n_neighbors=5)

classifer.fit(X_train,y_train)

y_pred = classifer.predict(X_test)





#confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)