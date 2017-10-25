import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from preprocess import preprocess_train, preprocess_test, featPCA, printSkew, preprocess_train_lda
from preprocess import preprocess_test_lda, getSmote, upsample, downsample, preprocess_train_svc, preprocess_test_svc 
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter
from sklearn.svm import SVC

#Get base accuracy
#all_zero = np.zeros(train['id'].count())
#base = np.absolute(y - all_zero)

#import the training set

# splits for validation
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs= -1)
    grid_search.fit(X, y)
    grid_search.best_params_
    print(grid_search.best_params_, grid_search.best_score_)

def getBase(y):
	all_zero = np.zeros(len(y))
	print("If all 0 predicted accuracy = " + str(accuracy_score(y, all_zero)) + "\n")

cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

param_test1 = {
 'min_child_weight':[3,4,5,6,7]}

ind_params1 = {'n_estimators': 180, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic','nthread':16, 'max_depth' :5,
             'learning_rate':0.1, 'subsample':0.5, 'min_child_weight':3,
             'gamma':0}

ind_params2 = {'n_estimators': 100, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic','nthread':16, 'max_depth' :5,
             'learning_rate':0.1, 'subsample':0.5, 'min_child_weight':3,
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

def dosmoteKfod(X, y, model):
	kf = StratifiedKFold(n_splits=5, random_state=1)
	for train_idx, test_idx in kf.split(X,y):
		X_train, y_train = getSmote(X.iloc[train_idx], y.iloc[train_idx])
		model.fit(X_train, y_train)
		y_pred = model.predict(X.iloc[test_idx])
		print(accuracy_score(y.iloc[test_idx], y_pred))
		print(confusion_matrix(y.iloc[test_idx], y_pred))

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
def combineModels(X, y, X_test, models, k=False):
	models = {}
	model_df = pd.DataFrame(0, index=X_test.index, columns=['xgb'])
	if k == True:
		kf = StratifiedKFold(n_splits=5, random_state=1)
		for train_idx, test_idx in kf.split(X,y): 
			for key, model in models.iteritems():
				model.fit(X.iloc[train_idx], y.iloc[train_idx])
				model_df[key].iloc[test_idx] = model.predict(X.iloc[test_idx])
	else:
		for key, model in models.iteritems():
			model.fit(X, y)
			model_df[key] = model.predict(X_test)
	#combined = pd.concat([X_test, model_df])
	return model_df

def createCSV(pred, final):
	final['Y'] = pred
	final.to_csv(path_or_buf='final.csv', index=False)
	return

def LogisticCV(X, y):
	lr = LogisticRegression()
	grid_values = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
	# l2 best is 18
	model_lr = GridSearchCV(lr, param_grid=grid_values,
								scoring = 'accuracy', cv = 5, n_jobs= -1)
	model_lr.fit(X, y)
	print(model_lr.best_params_, model_lr.best_score_)

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

def validate(X, y, model):
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	print(accuracy_score(y_test, pred))
	print(confusion_matrix(y_test, pred))
	print(f1_score(y_test, pred))

def validateSmote(X, y, model):
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)
	columns = X_train.columns
	X_train, y_train = getSmote(X_train, y_train)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	print(accuracy_score(y_test, pred))
	print(confusion_matrix(y_test, pred))
	print(f1_score(y_test, pred))
	#for name, score in zip(X_train.columns, model.feature_importances_*10):
	#	print(name, score)

def validateUpSample(X, y, model):
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)
	columns = X_train.columns
	X_train, y_train = upsample(X_train, y_train)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	print(accuracy_score(y_test, pred))
	print(confusion_matrix(y_test, pred))
	print(f1_score(y_test, pred))

def validateDownSample(X, y, model):
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)
	columns = X_train.columns
	X_train, y_train = downsample(X_train, y_train)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	print(accuracy_score(y_test, pred))
	print(confusion_matrix(y_test, pred))
	print(f1_score(y_test, pred))

if __name__ == "__main__":
	
	#import the training set
	#X, y, lda = preprocess_train_lda()
	#X_test, final = preprocess_test_lda(a_lda=lda)
	#X, y, a_lda = preprocess_train_svc()
	#X_test, final = preprocess_test_svc(a_lda)
	#X, y = preprocess_train()
	#X_test, final = preprocess_test()
	#X = XX
	#y = yy
	#X_test = XX_test
	#model = KNeighborsClassifier(n_neighbors=20)
	#model = RandomForestClassifier(n_estimators=200, max_leaf_nodes=3, n_jobs=-1)
	#model = DecisionTreeClassifier()
	#model = xgb.XGBClassifier(**ind_params1)
	#model = LogisticRegression(penalty='l2', C= 1)
	#model = LinearSVC(C=0.1)
	#validateUpSample(X, y, model)
	#validateSmote(X, y, model)
	#validateDownSample(X, y, model)
	#dosmoteKfod(X, y, model)
	#X, y = getSmote(X, y)
	#X, y = upsample(X, y)
	#X, y = downsample(X, y)
	#model.fit(X,y)
	#pred = model.predict(X_test)
	#print(len(pred[pred == 1]))
	#createCSV(pred, final)
	#validate(X, y, model)
	#model.fit(X, y)
	#pred = model.predict(X_test)
	#print(model.coef_)
	#for name, score in zip(X.columns, model.feature_importances_*10):
	#	print(name, score)
	#doKfold(X, y, model)
	#print(len(pred[pred == 1]))
	#createCSV(pred, final)
	#for name, score in zip(X.columns, (rnd_clf.feature_importances_*10)):
	#	print(name,  str(round(score, 5)))

	#import the training set

	X, y, a_lda = preprocess_train_svc()
	X_test, final = preprocess_test_svc(a_lda)
	X_xgb, y_xgb, lda = preprocess_train_lda(with_pca=True)
	X_test_xgb, final = preprocess_test_lda(a_lda=lda)
	X2, y2 = preprocess_train()
	X_test2, final = preprocess_test()

	X_xgbs, y_xgbs = getSmote(X_xgb, y_xgb)
	Xs, ys = getSmote(X, y)
	X2s, y2s = getSmote(X2, y2)

	xgb_model1 = xgb.XGBClassifier(**ind_params1)
	xgb_model2 = xgb.XGBClassifier(**ind_params2)
	log_model = LogisticRegression()
	print('Start fit')
	xgb_model1.fit(X_xgbs,y_xgbs)
	xgb_model2.fit(X2s,y2s)
	log_model.fit(Xs,ys)
	print('Start first predict')
	xgb_pred_init1 = pd.DataFrame(xgb_model1.predict(X_xgbs), columns = ['XGB'])
	xgb_pred_init2 = pd.DataFrame(xgb_model2.predict(X2s), columns = ['XGB2'])
	log_pred_init = pd.DataFrame(log_model.predict(Xs), columns = ['log'])

	train_this = pd.concat([log_pred_init, xgb_pred_init1, xgb_pred_init2], axis =1)
	print('Start train')
	fin_mod = LogisticRegression()
	fin_mod.fit(train_this,ys)
	print('Start next predict')
	xgb_pred_test1 = pd.DataFrame(xgb_model1.predict(X_test_xgb), columns = ['XGB'])
	xgb_pred_test2 = pd.DataFrame(xgb_model2.predict(X_test2), columns = ['XGB2'])
	log_pred_test = pd.DataFrame(log_model.predict(X_test), columns = ['log'])
	print('Almost done')
	fin_pred = pd.concat([log_pred_test, xgb_pred_test1, xgb_pred_test2], axis=1)
	pred = fin_mod.predict(fin_pred)

	print(len(pred[pred == 1]))
	createCSV(pred, final)
