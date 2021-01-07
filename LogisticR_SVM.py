#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:48:21 2020

@author: shriya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from scipy import stats
import warnings
import pickle
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

music_data = pd.read_csv('Data/data.csv')
music_data = music_data.drop('filename', 1)
music_data.label = pd.Categorical(music_data.label)
music_data['label'] = music_data.label.cat.codes 
X = music_data.drop(['label','beats','mfcc2','rolloff'],1)
Y = music_data['label']
X = preprocessing.scale(X)


Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=3)


## LOGISTIC WITH STRATIFYKFOLD SAMPLING
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
lin_clf = LogisticRegression(multi_class='ovr')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
rf_grid = GridSearchCV(estimator=lin_clf, param_grid=grid, cv=skf, n_jobs=-1)
rf_grid.fit(Xtrain, ytrain)
rf_params_max = rf_grid.best_params_
print("accuracy:")
print(rf_grid.score(Xtrain, ytrain))
print("params:")
print(rf_params_max)
print("")
rf_model = LogisticRegression(**rf_params_max)
rf_model.fit(Xtrain, ytrain)
rf_preds = rf_model.predict(Xtest)
print("validation accuracy")
print(accuracy_score(ytest, rf_preds))
print("")

logistic_strat = 'Weights/logistic_strat.sav'
pickle.dump(rf_model, open(logistic_strat, 'wb'))



## NORMAL SVM WITH STRATIFYKFOLD SAMPLING
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
lin_clf = svm.SVC()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
rf_grid = GridSearchCV(estimator=lin_clf, param_grid=tuned_parameters, cv=skf, n_jobs=-1)
rf_grid.fit(Xtrain, ytrain)
rf_params_max = rf_grid.best_params_
print("accuracy:")
print(rf_grid.score(Xtrain, ytrain))
print("params:")
print(rf_params_max)
print("")
rf_model = svm.SVC(**rf_params_max)
rf_model.fit(Xtrain, ytrain)
rf_preds = rf_model.predict(Xtest)
print("validation accuracy")
print(accuracy_score(ytest, rf_preds))
print("")

svm_strat = 'Weights/svm_strat.sav'
pickle.dump(rf_model, open(svm_strat, 'wb'))



## NORMAL SVM WITH KFOLD
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



rf_model = svm.SVC()
k_fold = KFold(n_splits=5, random_state=1, shuffle=True)
rf_grid = GridSearchCV(estimator=rf_model, param_grid=tuned_parameters, cv=k_fold, n_jobs=-1)
rf_grid.fit(Xtrain, ytrain)
rf_params_max = rf_grid.best_params_
print("accuracy:")
print(rf_grid.score(Xtrain, ytrain))
print("params:")
print(rf_params_max)
print("")
rf_model = svm.SVC(**rf_params_max)
rf_model.fit(Xtrain, ytrain)
rf_preds = rf_model.predict(Xtest)
print("validation accuracy")
print(accuracy_score(ytest, rf_preds))
print("")

svm_kfold = 'Weights/svm_kfold.sav'
pickle.dump(rf_model, open(svm_kfold, 'wb'))





## LOGISTIC REGRESSION WITH KFOLD
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
model = LogisticRegression(multi_class='ovr')
cv = KFold(n_splits=5, shuffle=True, random_state=1)
rf_grid = GridSearchCV(estimator=model, param_grid=grid, cv=cv, n_jobs=-1)
rf_grid.fit(Xtrain, ytrain)
rf_params_max = rf_grid.best_params_
print("LR accuracy:")
print(rf_grid.score(Xtrain, ytrain))
print("LR params:")
print(rf_params_max)
print("")
rf_model = LogisticRegression(**rf_params_max)
rf_model.fit(Xtrain, ytrain)
rf_preds = rf_model.predict(Xtest)
print("LR validation accuracy")
print(accuracy_score(ytest, rf_preds))
print("")

logistic_kfold = 'Weights/logistic_kfold.sav'
pickle.dump(rf_model, open(logistic_kfold, 'wb'))


