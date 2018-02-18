import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def xgboost_predict(train,labels,test):
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

	num_rounds = 10000 #since eta is small, we need to have large number of iterations.
	xgtest = xgb.DMatrix(test)

	#create training and validation sets 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plist, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	print("training the first model")
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#reverse the training data and labels and use the last 5k for validation set. 
	# This is for building the second model with a different validation set. Helps improve the results marginally.
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plist, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	print("training the second model")
	preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


	
	#Averaging the predictions of the two xgboost models.
	preds = (preds1)*0.5 + (preds2)*0.5
	return preds

#reading training and test data
train  = pd.read_csv('libertymutual_train.csv', index_col=0)
test  = pd.read_csv('libertymutual_test.csv', index_col=0)
print("just read train and test data files")


labels = train.Hazard #Assign hazard scores as labels for xgboost training.
train.drop('Hazard', axis=1, inplace=True) #remove labels from the training data.

train_s = train #copying the data for np.array coversion
test_s = test

columns = train.columns
test_ind = test.index


train_s = np.array(train_s)
test_s = np.array(test_s)

# change the categorical values of features into numerical values using label encoder.
for i in range(train_s.shape[1]): #run this for each feature.
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])


train_s = train_s.astype(float)
test_s = test_s.astype(float)

preds = xgboost_predict(train_s,labels,test_s) 

print("final predition time")

#print the predictions to a csv file.
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_two_models.csv')