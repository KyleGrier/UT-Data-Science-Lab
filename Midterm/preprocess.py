import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, zscore
from sklearn.preprocessing import StandardScaler, scale
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Fill the NaN locations for the columns given with the
# replacement values given
def fixNaN(df, cols, replace):
	fixer = zip(cols, replace)
	for col, rep in fixer :
		df[col] = df[col].fillna(rep)
	return df

# Remove columns in df based on the labels in to_remove
def removeCols(df, to_remove):
	#to_remove = ['F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F21', 'F24']
	df = df.drop(to_remove, 1)
	return df

# Get the features and labels for the training set
def getXy(df):
	y = df['Y']
	X = df.loc[:,'F1':'F27']
	return X, y

# Get the features and id to create final submission
def getXid(df):
	final = pd.DataFrame(df['id'])
	X = df.loc[:,'F1':'F27']
	return X, final

# Show all features that are largely the same value
def showCounts(X):
	feat_percent = {}
	len_feat = len(X)
	for col in X.columns:
		feat_count = Counter(X[col].values)
		largest = float(feat_count.most_common(1)[0][1])
		feat_percent[col] = largest/len_feat
	return feat_percent

# replace missing values using regression
def regReplace(X_other, X_miss):
	model = LinearRegression()
	test_idx = X_miss.isnull()
	train_idx = ~test_idx
	x_miss_train = X_miss[train_idx]
	x_miss_test = X_miss[test_idx]
	x_other_train = X_other[train_idx]
	x_other_test = X_other[test_idx]
	model.fit(x_other_train, x_miss_train)
	X_miss[test_idx] = model.predict(x_other_test)
	print(X_miss.describe())
	return X_miss

def removeOutliers(X,y):
	keep = (np.abs(zscore(X)) < 6).all(axis=1)
	X = X[keep]
	y = y[keep]
	return X, y

# Scale features
def toscale(X):
	scaler = StandardScaler()
	X_scale = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
	for col in X_scale.columns:
		X_scale[col] = scaler.fit_transform(np.sqrt(np.log1p(X[col] + abs(X[col]))).values.reshape(-1,1))
		#X_scale[col] = np.log1p(X[col] + abs(X[col]))
	return X_scale

def printSkew(X):
	skewness = {}
	for col in X.columns:
 		skewness[col] = skew(X[col])
 	print(str(skewness) + '\n')

def featPCA(X, keep, unique, verbose=False):
	pca = PCA()
	X_reduced = pca.fit_transform(X)
	#print(X_reduced.shape)
	if verbose:
		print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
	col_names = []
	for i in range(keep):
		col_names.append('pca_'+str(i)+unique)
	return pd.DataFrame(X_reduced[:,:keep],columns=col_names)

def getColumns(X, cols):
	X_other = X.loc[:, cols]
	return X_other

def dolda(X, y, unique):
	lda = LinearDiscriminantAnalysis()
	#X, y = getSmote(X, y)
	pred = lda.fit_transform(X, y)
	return pd.DataFrame(pred, columns=['lda'+unique]), lda

def getPoly(X, degree, unique):

	poly_pre = PolynomialFeatures(degree=degree).fit_transform(X)
	columns = []
	for i in range(poly_pre.shape[1]):
		columns.append(unique + str(i))
	poly = pd.DataFrame(PolynomialFeatures(degree=degree).fit_transform(X),columns=columns)
	poly = poly.drop(columns[0], 1)
	return poly

def getSmote(X, y):
	X_resampled, y_resampled = SMOTE().fit_sample(X, y)
	X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
	y_resampled = pd.DataFrame(y_resampled.ravel(), columns=[y.name])
	return X_resampled, y_resampled

def upsample(X, y):
	df = pd.concat([X,y], axis=1)
	df_0 = df[df['Y']==0]
	df_1 = df[df['Y']==1]
	df_1 = resample(df_1, replace=True, n_samples=len(df_0), random_state=123)
	df = pd.concat([df_0, df_1])
	X = df.drop(['Y'], 1)
	y = pd.DataFrame(df.Y, columns=[y.name])
	return X, y

def downsample(X, y):
	df = pd.concat([X,y], axis=1)
	df_0 = df[df['Y']==0]
	df_1 = df[df['Y']==1]
	df_0 = resample(df_0, replace=False, n_samples=len(df_1), random_state=123)
	df = pd.concat([df_0, df_1])
	X = df.drop(['Y'], 1)
	y = pd.DataFrame(df.Y, columns=[y.name])
	return X, y

# Do preprocessing for the training set
def preprocess_train_lda(with_pca=False):
	train = pd.read_csv("train_final.csv")
	X, y = getXy(train)
	cols = ['F5', 'F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F21', 'F24')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14', 'F26')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = toscale(X)
	X_best = getColumns(X, best_cols)
	X_best, lda = dolda(X_best, y, 'best')
	#X_best = getPoly(X_best, 2, 'best')
	X = removeCols(X, list(worst_cols))
	X = removeCols(X, list(best_cols))
	X = featPCA(X, 6, 'better', verbose=True)
	X = pd.concat([X, X_best], axis=1)
	return X, y, lda

# Do preprocessing for the test set
def preprocess_test_lda(a_lda = None):
	test = pd.read_csv("test_final.csv")
	X, final = getXid(test)
	# cols and replace have to be same length
	cols = ['F5','F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F21', 'F24')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14', 'F26')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = toscale(X)
	X_best = getColumns(X, best_cols)
	if a_lda:
		X_best = pd.DataFrame(a_lda.transform(X_best), columns=['ldabest'])
	else:
		X_best = featPCA(X_best, 1, 'best')
	#X_best = getPoly(X_best, 2, 'best')
	X = removeCols(X, list(worst_cols))
	X = removeCols(X, list(best_cols))
	X = featPCA(X, 6, 'better', verbose=True)
	X = pd.concat([X, X_best], axis=1)
	return X, final

# Do preprocessing for the training set
def preprocess_train():
	train = pd.read_csv("train_final.csv")
	X, y = getXy(train)
	cols = ['F5', 'F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F24')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14', 'F26')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = removeCols(X, list(worst_cols))
	X = toscale(X)
	X = featPCA(X, 12, 'better', verbose=True)
	#X, y = upsample(X, y)
	#X = getPoly(X, 2, 'best')
	return X, y

# Do preprocessing for the test set
def preprocess_test():
	test = pd.read_csv("test_final.csv")
	X, final = getXid(test)
	# cols and replace have to be same length
	cols = ['F5','F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F24')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14', 'F26')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = removeCols(X, list(worst_cols))
	X = toscale(X)
	X = featPCA(X, 12, 'better', verbose=True)
	return X, final

# Do preprocessing for the training set
def preprocess_train_svc():
	train = pd.read_csv("train_final.csv")
	X, y = getXy(train)
	cols = ['F5', 'F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	#worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F21', 'F24')
	worst_cols = ('F1','F4','F7','F8','F12','F13')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14')
	#best_cols = ('F3', 'F23')
	next_cols = ('F3', 'F23', 'F25', 'F26', 'F19')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = toscale(X)
	X_best = getColumns(X, best_cols)
	#X_next = getColumns(X, next_cols)
	#X_worst = getColumns(X, worst_cols)
	#X_worst = featPCA(X_worst, 4, 'worst', verbose=True)
	X_best, a_lda = dolda(X_best, y, 'best')
	#X_next, b_lda = dolda(X_next, y, 'next')
	X = pd.concat([X_best], axis=1)
	#X = getPoly(X, 2, 'best')
	return X, y, a_lda

# Do preprocessing for the test set
def preprocess_test_svc(a_lda):
	test = pd.read_csv("test_final.csv")
	X, final = getXid(test)
	# cols and replace have to be same length
	cols = ['F5','F19']
	replace = [0.0, 0]
	X = fixNaN(X, cols, replace)
	#worst_cols = ('F1','F4','F7','F8','F12','F13', 'F15', 'F17', 'F20', 'F21', 'F24')
	worst_cols = ('F1','F4','F7','F8','F12','F13')
	#best_cols = ('F2', 'F3', 'F14', 'F23', 'F25')
	best_cols = ('F2', 'F14')
	#best_cols = ('F3', 'F23')
	next_cols = ('F3', 'F23', 'F25', 'F26', 'F19')
	#corr_cols = ('F2', 'F14', 'F18','F25', 'F26')
	X = toscale(X)
	X_best = getColumns(X, best_cols)
	#X_next = getColumns(X, next_cols)
	#X_worst = getColumns(X, worst_cols)
	#X_worst = featPCA(X_worst, 4, 'worst', verbose=True)
	X_best = pd.DataFrame(a_lda.transform(X_best), columns=['ldabest'])
	#X_next = pd.DataFrame(b_lda.transform(X_next), columns=['next'])
	X = pd.concat([X_best], axis=1)
	#X = getPoly(X, 2, 'best')
	return X, final

#if __name__ == "__main__":
