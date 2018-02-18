# Regression-using-XGBoost
Property risk prediction using XGBoost

Libraries to be installed.
1. Pandas Data analysis library
2. Scikit learn
3. XGBoost for python
4. numpy 
5. matplotlib for plotting and pictorial representation of data and other data related graph plotting.


xgb_cv.py - cross validation code for tuning xgboost parameters in python.
xgbonly.py - python code for standalone xgboost regressor model with two xgboost models. This file evolved from one model to two models. I have just changed this file to build two models instead of one.
xgb.py - python code that generates the feature-importance.jpg file which ranks the features based on the fscore for each feature.
rf_cv.py - cross validation code for tuning RandomForestRegressor parameters in python.
rf.py - python code for standalone random forest regressor model.
ensemble.py - python code for ensemble of xgboost and RF regressor models.

xgb_feature_importance.png - Plot of all features based on their importance. Generated using XGBoost model in xgb.py.
