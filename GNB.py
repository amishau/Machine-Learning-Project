import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle

#Reading dataset
data = pd.read_csv('Data/data.csv')

#Separating into attribute values and labels
X = np.array(data.iloc[:,1:29])
y = np.array(data.iloc[:,29])

#Standardising the data
X = preprocessing.scale(X)

#Splitting into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Training the model
clf = GaussianNB()
clf.fit(X_train, y_train)

#Predicting labels and evaluating performance of model
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))

#Saving the model
file = open('Weights/GNB.pickle','wb')
pickle.dump(clf, file)
file.close()






