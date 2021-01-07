#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:47:10 2020

@author: shriya
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


music_data = pd.read_csv('Data/data.csv')
music_data = music_data.drop('filename', 1)
X = music_data.drop(['label','beats','mfcc2', 'rolloff'],1)
Y = music_data['label']


X = preprocessing.scale(X)

print (music_data.describe())

# Class countplot
s = sns.countplot(x='label', data=music_data)
figure = s.get_figure()    
figure.savefig('countplot.png', dpi=400)

# Generating feature array
genres = music_data.groupby('label')
features = list(music_data.columns)
features.remove('label')


# Plotting trends for each feature
for feat in features:
    sns.catplot(data=music_data, x='label', y=feat, jitter = True, dodge = True)

# features x features plot
snsplot = sns.pairplot(music_data)
snsplot.savefig('plot.png')

# Baseline model
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=1, stratify=Y)

logis = LogisticRegression(random_state=0, multi_class='auto').fit(Xtrain, ytrain.ravel())
pred_y = logis.predict(Xtest)
acc = accuracy_score(ytest, pred_y)
