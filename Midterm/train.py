import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from preprocess import preprocess_train, preprocess_test, featPCA, printSkew
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import time

#Get base accuracy
#all_zero = np.zeros(train['id'].count())
#base = np.absolute(y - all_zero)

#import the training set

# splits for validation
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)
def getBase(y):
	all_zero = np.zeros(len(y))
	print("If all 0 predicted accuracy = " + str(accuracy_score(y, all_zero)) + "\n")

cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

ind_params = {'n_estimators': 10000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic','nthread':16, 'max_depth' : 5,
             'learning_rate':0.1, 'subsample':0.4, 'min_child_weight':3}

def doGridCV(X,y):
	opt_xgb = GridSearchCV(xgb.XGBClassifier(**ind_params), 
	                            cv_params, 
	                             scoring = 'accuracy', cv = 5, n_jobs= -1)
	opt_xgb.fit(X, y)
	print(opt_xgb.grid_scores_)

def doKfold(X,y, model):
	kf = StratifiedKFold(n_splits=5, random_state=1)
	start = time.time()
	for train_idx, test_idx in kf.split(X,y):
	    model.fit(X.iloc[train_idx], y.iloc[train_idx])
	    y_pred = model.predict(X.iloc[test_idx])
	    print(accuracy_score(y.iloc[test_idx], y_pred))
	elapsed = time.time() - start
	print(elapsed)

def doKfoldXGB(X,y):
	kf = StratifiedKFold(n_splits=5, random_state=1)
	start = time.time()
	for train_idx, test_idx in kf.split(X,y):
	    model = xgb.XGBClassifier(**ind_params) 
	    model.fit(X.iloc[train_idx], y.iloc[train_idx])
	    y_pred = model.predict(X.iloc[test_idx])
	    print(accuracy_score(y.iloc[test_idx], y_pred))
	elapsed = time.time() - start
	print(elapsed)

def testCombine(X):
	model = LogisticRegression()
	return

def createXGB(X,y):
	model = xgb.XGBClassifier(**ind_params)
	model.fit(X,y) 
	return model

def createSVM(X,y):
	return

# If k = true do kfold to combine model
def combineModels(X, y, X_test, k=False):
	models = {}
	model_df = pd.DataFrame(0, index=X_test.index, columns=['xgb'])
	xgb_model = createXGB(X,y)
	models['xgb'] = xgb_model
	if k == True:
		kf = StratifiedKFold(n_splits=5, random_state=1)
		for train_idx, test_idx in kf.split(X,y): 
			for key, model in models.iteritems():
				model_df[key] = model.predict(X.iloc[test_idx])
	else:
		for key, model in models.iteritems():
			model_df[key] = model.predict(X_test)
	combined = pd.concat([X_test, model_df])
	return combined

def createCSV(pred, final):
	final['Y'] = pred
	final.to_csv(path_or_buf='final.csv', index=False)
	return



if __name__ == "__main__":
	#import the training set
	X, y = preprocess_train()
	getBase(y)
	X_test, final = preprocess_test()
	printSkew(X)
	print(X.describe())
	#X = featPCA(X)
	#X_test = featPCA(X_test)
	model = createXGB(X, y)
	print(model.feature_importances_)
	#model = LogisticRegression()
	#model = LinearSVC(random_state=0)
	#model.fit(X, y)
	pred = model.predict(X_test)
	print(len(pred[pred == 1]))
	createCSV(pred, final)


