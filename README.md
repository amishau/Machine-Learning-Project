# Music-Genre-Prediction

Music forms a very crucial part of our lives. We listen to and enjoy the music of different languages and genres. Music brings like-minded people together and is the glue that holds communities together. Communities can be recognized by the type/genre of songs that they compose, or even listen to.  The purpose of our project and research is to find an appropriate and accurate machine learning model, as well as preprocessing techniques to come up with an effective way of predicting the genre of any song for easy classification of music. This kind of issue is tackled by several music apps like Spotify, SoundCloud, Gaana, etc. which use predictive algorithms to suggest songs of their favourite music genres to the user.

### Folders
Weights (Saved Models), Data (Dataset)

### Files

**feature_selection.py**: Usage of correlation heatmap to obtain correlation values which are used for feature selection. Followed by an ExtraTreesClassifier for plotting the relationship between the importance of each feature with respect to the output layer which was useful for feature selection. The pie chart shows the number of samples in every class. Kernel density plots that highlight the distribution of observations.

**EDA.py**: Plots count plots to indicate class distribution in data, cat plots for each feature vs classes, features vs features plots, fitting of the baseline model

**RF.py**: Implements Random Forest followed by GridSearchCV used for optimizing the model and uses KFold Validation.

**LogisticR_SVM.py**: Implements Logistic Regression and Support Vector Machine Classifiers with both KFold and Stratified KFold cross validation techniques.

**KNN.py**: Implements K-Nearest Neighbours classifier with stratified K-fold cross validation technique.

**GNB.py**:  Implements Gaussian Naive Bayes classifier with stratified K-fold cross-validation technique.



