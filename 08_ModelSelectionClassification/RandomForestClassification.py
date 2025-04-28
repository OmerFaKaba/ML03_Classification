# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Verilerin Import Edilmesi
dataset = pd.read_csv("breast_cancer.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

#Train ve Test Setleri
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1)

#Feature Scaling (Özellik Ölçekleme)
#Standardization
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#Random Forest Modelinin Kurulması
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy",
                                    random_state=0)

classifier.fit(X_train, y_train)

#Modelin Test Set Üzerinde Denenmesi
y_pred = classifier.predict(X_test)


#Confusion Matrix ve Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
cmRF = confusion_matrix(y_test, y_pred)
AccuracyScoreRF = accuracy_score(y_test, y_pred)

#k-Fold Cross Validation Yöntemiyle Modelin Preformansının Ölçülmesi
from sklearn.model_selection import cross_val_score
accuriciesRF = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
AccuriciesMeanRF = accuriciesRF.mean()*100
StandartDeviationRF = accuriciesRF.std()*100































