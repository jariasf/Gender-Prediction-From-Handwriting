import numpy as np

def getRankings(clf, ranks, key):
	clf_ = clf[key]	
	for i in range(0,ranks,1):
		clf_.fit(X1, y1)
		feature_importance = clf_.feature_importances_
		saveToMatlabFile( key + '_imp_' + str(i), {'importance':feature_importance})

def getFeaturesImportance(clf, X , y):
	clf.fit(X, y)
	feature_importance = clf.feature_importances_
	return feature_importance

def retainFeaturesByImportance(feature_importance, X1, number_features):
	sorted_idx = np.argsort(feature_importance)[::-1]
	X1 = X1[:, sorted_idx[0:number_features]]
	return X1

def retainFeaturesByThreshold(feature_importance, X1, fi_threshold):
	important_idx = np.where(feature_importance > fi_threshold)[0]
	sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
	X1 = X1[:, important_idx][:, sorted_idx]
	return X1
