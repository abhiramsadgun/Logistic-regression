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

#Performance metrics

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

score=accuracy_score(y_test,y_ped)
print(score)
cm=confusion_matrix(y_test,y_ped)
print(cm)
print(classification_report(y_test,y_ped))

model=LogisticRegression()
c_values=[100,10,1,0.1,0.01]
param=[
    {'l1_ratio': [0.0, 1.0], 'C': c_values, 'solver': ['liblinear']},
    {'l1_ratio': [0.0, 0.5, 1.0], 'C': c_values, 'solver': ['saga']},
    {'l1_ratio': [0.0], 'C': c_values, 'solver': ['lbfgs', 'newton-cg', 'sag']}
]
#CV->cross validation we use GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold()
gsv=GridSearchCV(estimator=model,param_grid=param,scoring='accuracy',cv=cv,n_jobs=-1)
print(gsv)
gsv.fit(x_train,y_train)
print(gsv.best_params_)
print(gsv.best_score_)
y_ped=gsv.predict(x_test)

score=accuracy_score(y_test,y_ped)
print(score)
cm=confusion_matrix(y_test,y_ped)
print(cm)
print(classification_report(y_test,y_ped))


#Randomized search cv
from sklearn.model_selection import RandomizedSearchCV
model=LogisticRegression()
rs=RandomizedSearchCV(estimator=model,param_distributions=param,cv=5,scoring='accuracy',n_jobs=-1)
rs.fit(x_train,y_train)

print(rs.best_params_)
print(rs.best_score_)

y_ped=rs.predict(x_test)

score=accuracy_score(y_test,y_ped)
print(score)
cm=confusion_matrix(y_test,y_ped)
print(cm)
print(classification_report(y_test,y_ped))
