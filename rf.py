from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation

train  = pd.read_csv('libertymutual_train.csv')		# read the training data file
test  = pd.read_csv('libertymutual_test.csv')		# read the test data file
print("just read train and test data files")


labels = train.Hazard		#Hazard scores of the training dataset
train.drop('Hazard', axis=1, inplace=True)		#Just keep the features of the properties in training data.

train_s = train
test_s = test

columns = train.columns
test_ind = test.index

train_s = np.array(train_s)
test_s = np.array(test_s)

# change the categorical values of features into numerical values using label encoder.
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)

rfclf = RandomForestRegressor(n_estimators=1200, max_features=11, n_jobs=-1, verbose=2) #best CV value was 13 for 1k est, 12 for 1.5k, 

rfclf.fit(train_s,labels)
preds = rfclf.predict(test_s)

preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_RF.csv')