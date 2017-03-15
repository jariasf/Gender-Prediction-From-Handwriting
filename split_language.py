import numpy as np

def splitDataByLanguage(X):
	arabic = []
	english = []
	arabicIndex = 0	
	for i in range(0,X.shape[0], 4):
		arabic.append(X[i,])
		arabic.append(X[i+1,])
		english.append(X[i+2,])
		english.append(X[i+3,])
	return [np.array(arabic), np.array(english)]


def getDataBasedOnIndex(X, indexes):
	df = X.loc[X['writer'].isin(indexes)]
	del df['language']
	del df['writer']
	y = df['Y'].values
	del df['Y']
	return [df.values, y]
