#This script is borrowed from kaggle kernel. The script just visualizes the feature importance based on the fscore calculated by the xgboost model.

import pandas as pd
import xgboost as xgb
import operator
from matplotlib import pylab as plt
import numpy as np 
from sklearn.feature_extraction import DictVectorizer


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat)) #creates a list of feature entries in the xgb.fmap file with the index, feature name.
        i = i + 1

    outfile.close()

def get_data():
    train = pd.read_csv("libertymutual_train.csv") #read the training data.

    features = list(train.columns[2:]) # initialize the set of features by excluding ID and hazard score columns of the training data.

    y_train = train.Hazard #y_train is a placeholder for the labels. All the hazard scores for training data is copied.
	
	#In this loop, the features without numerical values are assigned a numerical value based on the average of the hazard scores corresponding to that feature values. 
	#Example: if a feature has values 'A','B' and 'C'. All the entries with value 'A' are grouped together and the mean of their hazard scores is assigned to all the entries with value 'A'. Similarly other values are quantized.
	#This loop basically converts alphabetical values into numerical values. Just like label encoder.
    for feat in train.select_dtypes(include=['object']).columns:
        m = train.groupby([feat])['Hazard'].mean()
        train[feat].replace(m,inplace=True)

    x_train = train[features]

    return features, x_train, y_train


features, x_train, y_train = get_data()
ceate_feature_map(features)

xgb_params = {"objective": "reg:linear", "eta": 0.005, "max_depth": 9, "silent": 1}
num_rounds = 1000

dtrain = xgb.DMatrix(x_train, label=y_train)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

#importance of the feature is calculated based on fscore values of each feature.
importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

#Finally, the features are plotted as per their fscores and saved into a image.
plt.figure()
df.plot()
df.plot(kind='bar', x='feature', y='fscore', legend=False, figsize=(10, 15))
plt.title('Feature Importance')
plt.ylabel('Relative importance')
plt.gcf().savefig('xgb_feature_importance.png')