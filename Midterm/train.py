import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from preprocess import preprocess_train
from sklearn.model_selection import GridSearchCV
import time

#Get base accuracy
#all_zero = np.zeros(train['id'].count())
#base = np.absolute(y - all_zero)

X, y = preprocess_train()
#import the training set

# splits for validation
kf = StratifiedKFold(n_splits=5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)


#cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
#cv_params = {'max_depth': [3,5,7]}
cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 100, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic','nthread':16, 'max_depth':3}
start = time.time()
opt_xgb = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                            scoring = 'accuracy', cv = 5)

opt_xgb.fit(X, y)
elapsed = time.time() - start
print(elapsed)
print(opt_xgb.grid_scores_)



#XGBoost
'''
start = time.time()
for train_idx, test_idx in kf.split(X,y):
    model = xgb.XGBClassifier(**ind_params) 
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[test_idx])
    print(accuracy_score(y.iloc[test_idx], y_pred))
elapsed = time.time() - start
print(elapsed)

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
#cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}

opt_xgb = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = kf, n_jobs= -1)
opt_xgb.fit(X, y)
print(opt_xgb.grid_scores_)

for train_idx, test_idx in kf.split(X,y):
    dtrain = xgb.DMatrix()
    dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
    model = xgb.XGBClassifier(n_estimators=100, max_depth=9) 
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[test_idx])
    print(accuracy_score(y.iloc[test_idx], y_pred))
'''
