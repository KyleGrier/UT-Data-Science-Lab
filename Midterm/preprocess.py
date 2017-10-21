import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from collections import Counter

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
	df = df.drop(to_remove, 1)
	return df

# Get the features and labels for the training set
def getXy(df):
	y = df['Y']
	X = df.loc[:,'F1':'F27']
	return X, y

# Show all features that are largely the same value
def showCounts(X):
	feat_percent = {}
	len_feat = len(X)
	for col in X.columns:
		feat_count = Counter(X[col].values)
		largest = float(feat_count.most_common(1)[0][1])
		feat_percent[col] = largest/len_feat
	return feat_percent

# Scale features
def scale(X):
	scaler = StandardScaler()
	X_scale = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
	return X_scale

# Do preprocessing for the training set
def preprocess_train():
	train = pd.read_csv("train_final.csv")
	X, y = getXy(train)

	X = removeCols(X)
	# cols and replace have to be same length
	cols = ['F5']
	replace = [0.0]
	X = fixNaN(X, cols, replace)
	X = scale(X)
	return X, y


# Do preprocessing for the test set
def preprocess_test():
	return

#if __name__ == "__main__":
