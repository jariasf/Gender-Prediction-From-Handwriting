
import numpy as np

from sklearn.grid_search import ParameterGrid
from sklearn.metrics import confusion_matrix
from metrics import getKx2CVScores, getConfidences
from sklearn import base


def performGridSearchWithKx2CV(clf, X, y, params):

	grid = list(ParameterGrid(params))
	performances = []

	for p in grid:
		new_clf = base.clone(clf)
		new_clf.set_params(**p)
		performances.append(np.mean(getKx2CVScores(new_clf, X, y)[0]))
		print performances[-1]

	return [grid, performances]


def ROC( confidences, groundTruth, granularity = 0.1 ):

	threshold = 0.0
	TPR, FPR = [], []

	while( threshold <= 1.0 ):

		predicted = [ int(c > threshold) for c in confidences ]
		CM = confusion_matrix(groundTruth, predicted)
		TPR.append(float(CM[1,1]) / np.sum(CM[1,:]))
		FPR.append(float(CM[0,1]) / np.sum(CM[0,:]))

		threshold += granularity

	return (TPR, FPR)