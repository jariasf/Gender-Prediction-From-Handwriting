from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from myutils import cloneEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn import base
import numpy as np
from feature_selection import *
from split_language import *

def getConfidences( clf, X ):

	if 'predict_proba' in dir(clf):
		return clf.predict_proba(X)[:,1]

	confidences = clf.decision_function(X)
	confidences -= np.min(confidences)
	confidences /= np.max(confidences)
	
	return confidences

def getTestScores( groundtruth, predicted, confidences ):

	return [accuracy_score(groundtruth, predicted), roc_auc_score(groundtruth, confidences), log_loss(groundtruth, confidences)]

def getKx2CVScores( clf, X, y, k = 5):

	accuracies = np.zeros((k, 2))
	AUCs = np.zeros((k, 2))
	logLoss = np.zeros((k, 2))

	for i in xrange(k):

		clf12 = cloneEstimator(clf)
		clf21 = cloneEstimator(clf)

		X1, X2, y1, y2 = train_test_split(X, y, test_size = 0.5)
		
		clf12.fit(X1, y1)
		clf21.fit(X2, y2)

		confidences21 = getConfidences(clf21, X1)
		confidences12 = getConfidences(clf12, X2)

		y1_predicted = clf21.predict(X1)
		y2_predicted = clf12.predict(X2)

		accuracies[i,0] = accuracy_score(y1, y1_predicted)
		accuracies[i,1] = accuracy_score(y2, y2_predicted)

		confidences21 = getConfidences(clf21, X1)
		confidences12 = getConfidences(clf12, X2)

		AUCs[i,0] = roc_auc_score(y1, confidences21)
		AUCs[i,1] = roc_auc_score(y2, confidences12)

		logLoss[i,0] = log_loss(y1, confidences21)
		logLoss[i,1] = log_loss(y2, confidences12)

	return [accuracies, AUCs, logLoss]


def getKx2CVScoresByLanguage( clf, Xdf, indexes, selectfeatures = False, nfeatures = 0, clfFS = None, k = 5):

	accuracies = np.zeros((k, 2))
	AUCs = np.zeros((k, 2))
	logLoss = np.zeros((k, 2))

	for i in xrange(k):

		idx1,idx2 = train_test_split( indexes, test_size = 0.5)
		[X1,y1] = getDataBasedOnIndex(Xdf, list(idx1[:,0]))	
		[X2,y2] = getDataBasedOnIndex(Xdf, list(idx2[:,0]))	

		[X_arabic, X_english] = splitDataByLanguage(X1)
		[y_arabic, y_english] = splitDataByLanguage(y1)

		[X_arabic_test, X_english_test] = splitDataByLanguage(X2)
		[y_arabic_test, y_english_test] = splitDataByLanguage(y2)	

		if( selectfeatures == True):
			feature_importance_english = getFeaturesImportance(clfFS, X_english, y_english)
			X_english = retainFeaturesByImportance(feature_importance_english, X_english, nfeatures )
			X_english_test = retainFeaturesByImportance(feature_importance_english, X_english_test, nfeatures )
			feature_importance_arabic = getFeaturesImportance(clfFS, X_arabic, y_arabic)
			X_arabic = retainFeaturesByImportance(feature_importance_arabic, X_arabic, nfeatures )
			X_arabic_test = retainFeaturesByImportance(feature_importance_arabic, X_arabic_test, nfeatures )
		
		ytest21 = y_arabic
		ytest12 = y_english_test

		X1 = X_arabic
		y1 = y_arabic
		X2 = X_arabic_test
		y2 = y_arabic_test
		clf12 = base.clone(clf)
		clf21 = base.clone(clf)		
		clf12.fit(X1, y1)
		clf21.fit(X2, y2)
		confidences21 = getConfidences(clf21, X1)
		confidences12 = getConfidences(clf12, X2)
		
		X1 = X_english
		X2 = X_english_test
		y1 = y_english
		y2 = y_english_test
		clf12 = base.clone(clf)
		clf21 = base.clone(clf)		
		clf12.fit(X1, y1)
		clf21.fit(X2, y2)
		confidences21Eng = getConfidences(clf21, X1)
		confidences12Eng = getConfidences(clf12, X2)
		avg21 = (confidences21 + confidences21Eng)/2.0
		avg12 = (confidences12 + confidences12Eng)/2.0
		logLoss[i,0] = log_loss(ytest21, avg21)
		logLoss[i,1] = log_loss(ytest12, avg12)
		AUCs[i,0] = roc_auc_score(ytest21, avg21)
		AUCs[i,1] = roc_auc_score(ytest12, avg12)
		accuracies[i,0] = accuracy_score(ytest21, np.around(avg21))
		accuracies[i,1] = accuracy_score(ytest12, np.around(avg12))
	return [accuracies, AUCs, logLoss]

if __name__ == '__main__':
	
	pass
