import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import math
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv('Data/data.csv')
y = data['label']
x = data.drop('label', axis = 'columns')
X = x.drop('filename', axis = 'columns')
X = X.drop('beats', axis = 'columns')
X = X.drop('rolloff', axis = 'columns')
X = X.drop('mfcc2', axis = 'columns')

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = 1, stratify = y)

#MinMax Scaling
norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)

#Random Forest Classifier
rfc = RandomForestClassifier(max_features = int(math.sqrt(29)), n_estimators = 200)

cv1 = KFold(n_splits = 5, random_state = 1, shuffle = True)

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = cv1)
best_model = CV_rfc.fit(X_train, y_train)

#Predict target vector
pred = best_model.predict(X_test)

#Saving model in a pickle file
rf_kfold = 'Weights/rf_kfold.sav'
pickle.dump(best_model, open(rf_kfold, 'wb'))
print(accuracy_score(y_test, pred))
