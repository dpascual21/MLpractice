#By Drew_Pascual
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


dataset_url = ('http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')

data = pd.read_csv(dataset_url, sep=';')
print (data.head())
print (data.describe()) #summary statistics, count mean std etc

#Next: #Splitting the data into training and test sets at the beginning of your modeling workflow is crucial for getting a realistic estimate of your model's performance.
#Target feature: quality

y = data.quality
x = data.drop('quality', axis=1)

#sklearn train_test_split function, set aside 20% of data for test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123, stratify = y)

#scales the values for both the training and any new data, so the variance is centered around 0
scaler = preprocessing.StandardScaler().fit(X_train)

X_test_scaled = scaler.transform(X_test)

print (X_test_scaled.mean(axis=0))

print (X_test_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

print (pipeline.get_params())

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)

print (clf.refit)


print r2_score(y_test, y_pred)

print mean_squared_error(y_test, y_pred)


joblib.dump(clf, 'rf_regressor.pkl')

#clf2 = joblib.load('rf_regressor.pkl')

# Predict data set using loaded model
#clf2.predict(X_test)
