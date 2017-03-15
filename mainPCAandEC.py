from metrics import *
from myio import *
from tunning import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
from feature_selection import *
import matplotlib.pyplot as plt
import pandas as pd
from split_language import *
from feature_reduction import *
from sklearn import cross_validation
from EnsembleClassifier import *

if __name__ == '__main__':
	
	#################################I/O#################################
	print('Reading Files...')
	train = pd.read_csv('dataset/train.csv')
	test = pd.read_csv('dataset/test.csv')
	answers = pd.read_csv('dataset/train_answers.csv')
	features_labels = list(train.columns.values[5:]) 
	X = train[features_labels].values
	Xtest = test[features_labels].values
	y_ = answers['male'].values
	y = y_[0:200]
	ytest = y_[200:]
	y = np.repeat(y, 4 , axis=0) #Repeat 4 times because of writer
	ytest = np.repeat(ytest, 4 , axis=0)
	

	clf = {		# Parameters found with Grid search
		'LR'	:	LogisticRegression(C = 1000),
		'RF'	:	RandomForestClassifier(n_estimators = 500, max_features = 'sqrt', criterion = 'gini', n_jobs=-1),
		'LSVM'	:	LinearSVC( C = 1000 ),
		'SVM'	:	SVC(C = 100, gamma = 0.1, probability = True),
		'GB'	:	GradientBoostingClassifier(n_estimators = 1000, max_features = 'log2', loss = 'deviance', learning_rate = 0.1)
	}

	######################PCA###################
	'''
	pca_model = createPCAModel(X, 0.99, whiten = False)
	X = applyPCAToTrain(X, pca_model)
	Xtest = applyPCAToTest(Xtest, pca_model)
	
	###Cross Validation##
	for c in clf.keys():
		scores = getKx2CVScores(clf[c], X, y)
		print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[0]), np.std(scores[0]), c))
		print("LogLoss: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[2]), np.std(scores[2]), c))
		print("AUC: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[1]), np.std(scores[1]), c))
	
	###Real Testing###
	for c in clf.keys():
		clf[c].fit(X, y)
		y_predicted = clf[c].predict(Xtest)
		confidences = getConfidences(clf[c], Xtest)
		acc_score, auc_score, logloss_score = getTestScores(ytest, y_predicted, confidences)
		print c + ': ' + str(acc_score) + ' ' + str(auc_score) + ' ' + str(logloss_score)
	'''
	###########################  LANGUAGE PARTITION  #############################
	print('Partition of Dataset by languages...')
	[X_arabic, X_english] = splitDataByLanguage(X)
	[y_arabic, y_english] = splitDataByLanguage(y)

	[X_arabic_test, X_english_test] = splitDataByLanguage(Xtest)
	[y_arabic_test, y_english_test] = splitDataByLanguage(ytest)
	
	ytest = y_english_test
	
	####Feature selection####
	nfeatures = 100
	feature_importance_english = getFeaturesImportance(clf['GB'], X_english, y_english)
	X_english = retainFeaturesByImportance(feature_importance_english, X_english, nfeatures )
	X_english_test = retainFeaturesByImportance(feature_importance_english, X_english_test, nfeatures )	

	feature_importance_arabic = getFeaturesImportance(clf['GB'], X_arabic, y_arabic)
	X_arabic = retainFeaturesByImportance(feature_importance_arabic, X_arabic, nfeatures )
	X_arabic_test = retainFeaturesByImportance(feature_importance_arabic, X_arabic_test, nfeatures )

	########ENSEMBLE OF CLASSIFIERS#######
	eclf = EnsembleClassifier(clfs=[clf['RF'], clf['GB']], voting='soft')
	clf['ES'] = eclf
	
	'''
	####CROSS VALIDATION####
	features_labels.insert(0, train.columns.values[0])	
	features_labels.insert(1, train.columns.values[2]) #language for testing	
	Xdf = train[features_labels].copy()
	
	Xdf['Y'] = y
	indexes = answers.values[0:200]
	
	for c in clf.keys():
		scores = getKx2CVScoresByLanguage(clf[c], Xdf, indexes, True, nfeatures,  clf['GB'])
		print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[0]), np.std(scores[0]), c))
		print("LogLoss: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[2]), np.std(scores[2]), c))
		print("AUC: %0.3f (+/- %0.3f) [%s]" % (np.mean(scores[1]), np.std(scores[1]), c))
	'''
	
	###############################PREDICTION FOR EACH PARTITION###################################
	english_confidences = []
	from sklearn.metrics import accuracy_score
	print('English Prediction...')	
	for c in clf.keys():
		clf[c].fit(X_english, y_english)
		confidences = getConfidences(clf[c], X_english_test)		
		english_confidences.append( confidences )

	arabic_confidences = []
	print('Arabic Prediction...')
	for c in clf.keys():
		clf[c].fit(X_arabic, y_arabic)
		confidences = getConfidences(clf[c], X_arabic_test)
		arabic_confidences.append( confidences )
	
	i = 0
	for c in clf.keys():
		avg = (english_confidences[i] + arabic_confidences[i])/2.0
		logloss = log_loss(ytest, avg)
		print("Classifier %s, Accuracy: %0.3f, AUC: %0.3f, Logloss: %0.3f" % ( c, accuracy_score(ytest, np.around(avg)) , roc_auc_score(ytest, avg) ,  logloss ) )
		i = i + 1

