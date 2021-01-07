import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

#Extracting data and removing features
data = pd.read_csv('Data/data.csv')
y = data['label']
x = data.drop('label', axis = 'columns')
X = x.drop('filename', axis = 'columns')
X = X.drop('beats', axis = 'columns')
X = X.drop('rolloff', axis = 'columns')
X = X.drop('mfcc2', axis = 'columns')

#Correlation heatmap
correlations = data.corr()
fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(correlations, fmt='.2f',annot=True, cbar_kws={"shrink": .70})
plt.show()

#Extra trees classifier to check importance of features
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(28).plot(kind='barh')
plt.show()

#Pie chart showing class distributions
pie = data["label"].value_counts().plot.pie( autopct='%.2f', figsize=(10, 10),fontsize=20)
plt.show()

#Kernel desnity plot for visualizing the distributions of observations
plt.figure(figsize=(30,10))
sns.kdeplot(data=data.loc[data['label']=='reggae', 'spectral_bandwidth'], label="Reggae")
sns.kdeplot(data=data.loc[data['label']=='blues', 'spectral_bandwidth'], label="Blues")
sns.kdeplot(data=data.loc[data['label']=='jazz', 'spectral_bandwidth'], label="Jazz")
sns.kdeplot(data=data.loc[data['label']=='pop', 'spectral_bandwidth'], label="Pop")
sns.kdeplot(data=data.loc[data['label']=='country', 'spectral_bandwidth'], label="Country")
sns.kdeplot(data=data.loc[data['label']=='rock', 'spectral_bandwidth'], label="Rock")
sns.kdeplot(data=data.loc[data['label']=='metal', 'spectral_bandwidth'], label="Metal")
sns.kdeplot(data=data.loc[data['label']=='classical', 'spectral_bandwidth'], label="Classical")
sns.kdeplot(data=data.loc[data['label']=='hiphop', 'spectral_bandwidth'], label="Hiphop")
sns.kdeplot(data=data.loc[data['label']=='disco', 'spectral_bandwidth'], label="Disco")
plt.legend()
plt.show()
