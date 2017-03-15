import numpy as np
from sklearn import base
from metrics import getConfidences
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from myio import readFromMatlabFile

class CascadeClassifier:

	# classifier features (min,max)

	def __init__(self):

		self.classifiers = [
			GradientBoostingClassifier(n_estimators = 1000, max_features = 'log2', loss = 'deviance', learning_rate = 0.1),
			LogisticRegression(C = 1000),
			LinearSVC( C = 1000 ),
			LogisticRegression(C = 1000),
			GradientBoostingClassifier(n_estimators = 1000, max_features = 'log2', loss = 'deviance', learning_rate = 0.1)
			]
		self.features_indices = [ (readFromMatlabFile('GB_rank_sqrt.mat')['ranking'] - 1)[:50], range(5970,7066), range(950,5970), range(50,950), None ]
		self.limits = [(0.0001,0.9999), (0.2,0.8), (0.1,0.9), (0.1,0.9), None]

	def __getSelectedFeatures(self, X, i):

		if self.features_indices[i] == None:
			return X
		else:
			if len(X.shape) == 1:
				return X[self.features_indices[i]]
			return X[:,self.features_indices[i]]


	def fit(self, X, y):

		for i in xrange(len(self.classifiers)):
			self.classifiers[i].fit(self.__getSelectedFeatures(X, i), y)

	def predict(self, X):
		return [self.predict_single(X[i,:]) for i in xrange(X.shape[0])]

	def predict_single(self, x):

		for i in xrange(len(self.classifiers) - 1):
			confidence = getConfidences(self.classifiers[i], self.__getSelectedFeatures(x, i))

			if confidence < self.limits[i][0]:
				return 0
			if confidence > self.limits[i][1]:
				return 1

		return self.classifiers[-1].predict([x])[0]

	def predict_proba(self, X):
		return np.ones((X.shape[0], 2))

	def getClone(self):
		return CascadeClassifier()




class FSClassifier:

	def __init__(self, classifier, feature_selector, num_features = 100):
		self.classifier = classifier
		self.feature_selector = feature_selector
		self.num_features = num_features
		self.features = []

	def fit(self, X, y):

		self.feature_selector.fit(X, y)
		rank = np.argsort(-self.feature_selector.feature_importances_)
		self.features = rank[:self.num_features]

		return self.classifier.fit(X[:,self.features], y)

	def predict(self, X):
		return self.classifier.predict(X[:,self.features])
		

	def predict_proba(self, X):

		if 'predict_proba' in dir(self.classifier):
			return self.classifier.predict_proba(X[:,self.features])

		confidences = self.classifier.decision_function(X[:,self.features])
		confidences -= np.min(confidences)
		confidences /= np.max(confidences)

		probabilities = np.zeros((confidences.shape[0],2))
		probabilities[:,1] = confidences
		probabilities[:,0] = 1 - probabilities[:,0]

		return probabilities

	def getClone(self):
		return FSClassifier(base.clone(self.classifier), base.clone(self.feature_selector), self.num_features)
