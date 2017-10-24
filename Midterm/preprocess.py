import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, zscore
from sklearn.preprocessing import StandardScaler, scale
from collections import Counter
from sklearn.decomposition import PCA

# Fill the NaN locations for the columns given with the
# replacement values given
def fixNaN(df, cols, replace):
	fixer = zip(cols, replace)
	for col, rep in fixer :
		df[col] = df[col].fillna(rep)
	return df

# Remove columns in df based on the labels in to_remove
def removeCols(df):
	to_remove = ['F19', 'F8', 'F17', 'F24', 'F1', 'F4', 'F15', 'F7', 'F20', 'F12', 'F13']
	#to_remove = ['F25', 'F4', 'F17', 'F20']
	#to_remove =  ['F1', 'F4', 'F7', 'F8','F12','F13','F15', 'F17', 'F20', 'F24', 'F26', 'F23']
	#to_remove = ['F1', 'F2', 'F4', 'F5', 'F7', 'F8', 'F9', 'F10']
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
		#X_scale[col] = np.log1p(X_scale[col] + abs(X_scale[col]))
		X_scale[col] = np.log1p(X[col] + abs(X[col]))
	return X_scale

def printSkew(X):
	skewness = {}
	for col in X.columns:
 		skewness[col] = skew(X[col])
 	print(str(skewness) + '\n')

def featPCA(X, keep, unique):
	pca = PCA()
	X_reduced = pca.fit_transform(X)
	print(X_reduced.shape)
	print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
	col_names = []
	for i in range(keep):
		col_names.append('pca_'+str(i)+unique)
	return pd.DataFrame(X_reduced[:,:keep],columns=col_names)

def getColumns(X):
	X_other = X.loc[:, ('F1', 'F2', 'F4', 'F5', 'F7', 'F8', 'F9', 'F10', 'F12', 'F13', 'F14', 'F15', 'F17', 'F20', 'F24')]
	return X_other

# Do preprocessing for the training set
def preprocess_train():
	train = pd.read_csv("train_final.csv")
	X, y = getXy(train)
	cols = ['F5', 'F19']
	replace = [0.0, 6665.12]
	X = fixNaN(X, cols, replace)
	X = toscale(X)
	#X = removeCols(X)
	X = featPCA(X, 10, 'better')
	#X_rem = getColumns(X)
	#X = removeCols(X)
	#X_rem = featPCA(X_rem, 7,'worse')
	#X = featPCA(X, 7, 'better')
	#X = pd.concat([X, X_rem], axis=1)
	# cols and replace have to be same length
	#X, y = removeOutliers(X,y)
	return X, y


# Do preprocessing for the test set
def preprocess_test():
	test = pd.read_csv("test_final.csv")
	X, final = getXid(test)
	#X = removeCols(X)
	# cols and replace have to be same length
	cols = ['F5','F19']
	replace = [0.0, 6665.12]
	X = fixNaN(X, cols, replace)
	X = toscale(X)
	#X = removeCols(X)
	X = featPCA(X, 10, 'better')
	#X_rem = getColumns(X)
	#X = removeCols(X)
	#X_rem = featPCA(X_rem, 7, 'worse')
	#X = featPCA(X, 7, 'better')
	#X = pd.concat([X, X_rem], axis=1)
	return X, final

#if __name__ == "__main__":
