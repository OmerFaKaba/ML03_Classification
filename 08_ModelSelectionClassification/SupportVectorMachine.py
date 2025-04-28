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

#SVM Modelinin Kurulması
from sklearn.svm import SVC
classifier = SVC(probability=True, kernel="rbf")

#Modelin Eğitilmesi
classifier.fit(X_train, y_train)


#Test Set Üzerinde Sınama
y_pred = classifier.predict(X_test)
SVMProbability = classifier.predict_proba(X_test)

#Confusion Matrix ve Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cmSVM = confusion_matrix(y_test, y_pred)
AccuracyScoreSVM = accuracy_score(y_test, y_pred)

#k-Fold Cross Validation Yöntemiyle Modelin Preformansının Ölçülmesi
from sklearn.model_selection import cross_val_score
accuriciesSVM = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
AccuriciesMeanSVM = accuriciesSVM.mean()*100
StandartDeviationSVM = accuriciesSVM.std()*100


"""
#Train Set Sonuçlarının Görselleştirilmesi
from matplotlib.colors import ListedColormap
X_set = ss.inverse_transform(X_train)
y_set = y_train
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min() -1 , stop=X_set[:, 0].max() + 1, step=0.25),
                     np.arange(start =X_set[:, 1].min() -1 , stop=X_set[:, 1].max() + 1, step=0.25))

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Support Vector Machine - Train Set (Linear)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
"""






















