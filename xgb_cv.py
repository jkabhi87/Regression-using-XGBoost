import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005 # Values tried in CV .005 .01
params["min_child_weight"] = 6
params["subsample"] = 0.7 # Values tried in CV 0.6 0.65 0.7 0.75
params["colsample_bytree"] = 0.7 # Values tried in CV 0.6 0.65 0.7 0.75
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9 # Values tried in CV 6 8 9


plst = list(params.items()) #parameter list for xgboost model

train  = pd.read_csv('libertymutual_train.csv') #read training data
labels = train.Hazard #read labels
features = list(train.columns[2:]) #All the columns from 3 to the end represent features. First column is id and second is hazard score.

train.drop('Hazard', axis=1, inplace=True)

train_s = train #create a array representation of the training data using train_s

columns = train.columns

train_s = np.array(train_s)

# This loop is to change the features with categorical values into numerical representation. This quantifies the categorical features.
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])


train_s = train_s.astype(float)

train = xgb.DMatrix(train_s, label=labels)
num_rounds = 10000 #High value for num_rounds as the learning rate is small
#early stopping values tried 100, 110, 130, 150, 180. 150 yielded best results. There was no improvement in results from 150 to 180.
cvresult = xgb.cv(params=plst, dtrain=train, num_boost_round=num_rounds, nfold=5, verbose_eval=True, early_stopping_rounds=150)
