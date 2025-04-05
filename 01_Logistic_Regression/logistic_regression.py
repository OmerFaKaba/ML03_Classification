# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:02:25 2025

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


#feature scaling
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#Logistic Regression model setup
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)


classifier.fit(X_train,y_train)


#logistic regression model try
LogistricRegressionPredict = classifier.predict(ss.transform([[30,150000]]))

LogistricRegressionPredictProba = classifier.predict_proba(ss.transform([[30,150000]]))



#logistic regression test set predict
y_pred = classifier.predict(X_test)




#Model Accuracy
#Confusion Matrix & Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acur = accuracy_score(y_test, y_pred)


#visiualization of train set
from matplotlib.colors import ListedColormap
X_set = ss.inverse_transform(X_train)
y_set = y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-10,stop =X_set[:,0].max()+ 10,step=0.25 ),
                     np.arange(start=X_set[:,1].min()-1000,stop=X_set[:,1].max()+1000,step=0.25 ))

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression - Train Set')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
















