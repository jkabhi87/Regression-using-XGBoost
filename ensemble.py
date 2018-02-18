import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
import operator

def xgboost_pred(train,labels,test):
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.005
	params["min_child_weight"] = 6
	params["subsample"] = 0.7
	params["colsample_bytree"] = 0.7
	params["scale_pos_weight"] = 1
	params["silent"] = 1
	params["max_depth"] = 9
    
    
	plist = list(params.items())

	#Using 5000 rows for validation data set.  
	offset = 5000

	num_rounds = 10000
	xgtest = xgb.DMatrix(test)

	#create training and validation sets 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#train using early stopping and predict. Early stopping helps terminate the loop early and avoid unnecessary iterations. Loop is quit when there is no improvement in the test error after the specified number of iterations
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plist, xgtrain, num_rounds, watchlist, early_stopping_rounds=150)
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#reverse the training data and labels and use the last 5k for validation set. 
	# This is for building the second model with a different validation set. Helps improve the results marginally.
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plist, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#combine predictions
	#since the metric only cares about relative rank we don't need to average
	preds = (preds1)*0.5 + (preds2)*0.5
	return preds

#load train and test 
train  = pd.read_csv('libertymutual_train.csv', index_col=0)
test  = pd.read_csv('libertymutual_test.csv', index_col=0)
print("just read train and test data files")


labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

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

preds1 = xgboost_pred(train_s,labels,test_s)

#now build a RF regressor model and predict.
rfclf = RandomForestRegressor(n_estimators=1000, max_features=13, n_jobs=-1, verbose=2) #0.376178 for 13 , 0.378154 for sqrt
rfclf.fit(train_s,labels)
preds2 = rfclf.predict(test_s)

#Assign the average of both the predictions as the final hazard score.
preds = (preds1)*0.5 + (preds2)*0.5

print("final predition time")


#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_RF_ensemble.csv')
