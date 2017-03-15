from metrics import *
from myio import *
from tunning import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier
from classifier import CascadeClassifier
from metrics import getConfidences, getTestScores
import numpy as np

if __name__ == '__main__':
	
	print('Reading Files...')
	labels = getLabelsFromCSVFile( 'dataset/train_answers.csv' )
	[X_, y] = getDatasetFromCSVFile( 'dataset/train.csv', labels )
	X = X_[:,4:]

	tortuosity = range(40)					# 2
	directionPerpendicular = range(40,50)	# 1
	curvature = range(50,950)				# 4 Maybe LSVM
	chainCode = range(950,5970)				# 3 Maybe LogR
	directions = range(5970,7066)			# 5
	
	[Xtest_, ytest] = getDatasetFromCSVFile( 'dataset/test.csv', labels )
	Xtest = Xtest_[:,4:]

	clf = {		# Parameters found with Grid search
		'LR'	:	LogisticRegression(C = 1000),
		'RF'	:	RandomForestClassifier(n_estimators = 500, max_features = 'sqrt', criterion = 'gini', n_jobs = 2),
		'LSVM'	:	LinearSVC( C = 1000 ),
		'SVM'	:	SVC(C = 100, gamma = 0.1, probability = True),
		'GB'	:	GradientBoostingClassifier(n_estimators = 1000, max_features = 'log2', loss = 'deviance', learning_rate = 0.1)
		#'CC' : CascadeClassifier()
		#'FS' : FSClassifier()
	}

	print('Baseline Prediction...')
	for c in clf.keys():
		clf[c].fit(X, y)
		y_predicted = clf[c].predict(Xtest)
		confidences = getConfidences(clf[c], Xtest)
		acc_score, auc_score, logloss_score = getTestScores(ytest, y_predicted, confidences)
		print("Classifier %s, Accuracy: %0.3f, AUC: %0.3f, Logloss: %0.3f" % ( c, acc_score , auc_score , logloss_score ) )

	'''	
	####### CROSS VALIDATION #####	
	for c in clf.keys():
		scores = getKx2CVScores(clf[c], X, y)
		print c + ':\t' + str(np.mean(scores[0])) + ' ' + str(np.std(scores[0])) + ' ' + str(np.mean(scores[1])) + ' ' + str(np.std(scores[1])) + ' '	+ str(np.mean(scores[2])) + ' ' + str(np.std(scores[2]))
	'''
	

	

	
