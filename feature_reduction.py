from sklearn.decomposition import PCA

def createPCAModel(X, k, whiten = True):
	if(k == -1):
		pca_model = PCA()
	else:
		pca_model = PCA(n_components=k, whiten = whiten )
	pca_model.fit(X)
	return pca_model

def applyPCAToTrain(X, pca_model):
	pca = pca_model.fit_transform(X)
	return pca

def applyPCAToTest(X, pca_model):
	pca = pca_model.transform(X)
	return pca
