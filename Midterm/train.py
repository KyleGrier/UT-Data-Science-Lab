import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from preprocess import preprocess_train, preprocess_test, featPCA, printSkew
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

param_test1 = {
 'min_child_weight':[6,8,10,12]}

ind_params = {'n_estimators': 100, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic','nthread':16, 'max_depth' : 5,
             'learning_rate':0.1, 'subsample':0.8, 'min_child_weight':3,
             'gamma':0}

def doGridCV(X,y):
	opt_xgb = GridSearchCV(xgb.XGBClassifier(**ind_params), 
	                            param_test1, 
	                            scoring = 'accuracy', cv = 5, n_jobs= -1)
	opt_xgb.fit(X, y)
	print(opt_xgb.best_params_, opt_xgb.best_score_)

def modelfit(X, y, cv_folds=5, early_stopping_rounds=50):
	alg = xgb.XGBClassifier(**ind_params)
	xgtrain = xgb.DMatrix(X.values, label=y.values)
	cvresult = xgb.cv(ind_params, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='error', early_stopping_rounds=early_stopping_rounds, stratified=True)
	print(cvresult.shape[0])
	print(cvresult)
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

def LogisticCV(X, y):
	lr = LogisticRegression()
	grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
	model_lr = GridSearchCV(lr, param_grid=grid_values,
								scoring = 'accuracy', cv = 5, n_jobs= -1)
	model_lr.fit(X, y)
	print(model_lr.grid_scores_)
#See the changes due to LDA
def plotLDA(X, y):
	clf = LinearDiscriminantAnalysis()

	label1 = X[y==0]
	label2 = X[y==1]

	clf.fit(X, y)

	label1_sk = clf.transform(label1)
	label2_sk = clf.transform(label2)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.scatter(label1_sk, np.zeros(len(label1_sk)), c='r', marker='o', label="label1")
	ax.scatter(label2_sk, np.ones(len(label2_sk)), c='b', marker='x', label="label2")
	ax.legend()
	plt.show()

if __name__ == "__main__":
	#import the training set
	X, y = preprocess_train()
	getBase(y)
	X_test, final = preprocess_test()
	printSkew(X)
	#plotLDA(X,y)
	doGridCV(X, y)
	#LogisticCV(X, y)
	#model = createXGB(X, y)
	#print(model.feature_importances_)
	#model = LogisticRegression(penalty='l2', C= 0.1)
	#model = LinearSVC(random_state=0)
	#model.fit(X, y)
	#pred = model.predict(X_test)
	#print(len(pred[pred == 1]))
	#createCSV(pred, final)


