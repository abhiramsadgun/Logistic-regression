from pandas.core.common import random_state
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

#Standarization is already done
x,y=make_classification(n_samples=500,n_features=10,n_classes=2,random_state=42)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.32,random_state=42)
 
from sklearn.linear_model import LogisticRegression
ls=LogisticRegression()
ls.fit(x_train,y_train)
y_ped=ls.predict(x_test)
print(y_ped)
print(ls.predict_proba(x_test))
plt.scatter(y_test,y_ped)
plt.xlabel("Actual")
plt.ylabel("predicted")
plt.show()
#Performance metrics

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

score=accuracy_score(y_test,y_ped)
print(score)
cm=confusion_matrix(y_ped,y_test)
print(cm)
print(classification_report(y_test,y_ped))
