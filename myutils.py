from sklearn import base
from sklearn.preprocessing import StandardScaler

def cloneEstimator( estimator ):

	if type(estimator).__name__ == 'instance':
		return estimator.getClone()
	else:
		return base.clone(estimator)

	return None

def createZScoreModel(X):
	scaler = StandardScaler()
	scaler.fit_transform(X)
	return scaler
	

def applyZScore(X, scaler):
	X = scaler.fit_transform(X)
	return X

