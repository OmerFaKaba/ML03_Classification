# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:16:18 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("breast_cancer.csv")

X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1] 


#train ve test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)



#Logistic Regression Setup
from sklearn.linear_model import LogisticRegression

classifer = LogisticRegression(random_state=0)


classifer.fit(X_train, y_train)

y_pred = classifer.predict(X_test)
y_pred_prob = classifer.predict_proba(X_test)


#Confusion Matrix and AccuracyScore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)
Accuracy = accuracy_score(y_test, y_pred)




#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuricies = cross_val_score(estimator = classifer, X=X,y=y,cv=10)
accuriciesMean = accuricies.mean()*100

StandartDEviation = accuricies.std()*100







