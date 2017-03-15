import scipy.io as sio
import numpy as np
import csv

def getDatasetFromCSVFile( path, labels ):

	reader = csv.reader(open(path, 'rb'))

	X = []
	y = []

	row_num = -1
	for row in reader:
		row_num += 1

		if row_num == 0:
			continue

		v = np.zeros(len(row))

		v[0] = float(row[0])						# writerID
		v[1] = float(row[1])						# pageID
		v[2] = 1.0 if row[2] == 'English' else 0.0	# Language (Is it in english?)
		v[3] = float(row[3])						# Same page?
		v[4:] = np.array(row[4:])
		
		X.append(v)
		y.append(labels[int(row[0])])

	return [np.array(X), np.array(y)]

def getLabelsFromCSVFile( path ):

	reader = csv.reader(open(path, 'rb'))
	labels = {}

	row_num = -1
	for row in reader:

		row_num += 1
		if row_num == 0:
			continue

		labels[int(row[0])] = int(row[1])

	return labels

def saveToMatlabFile( path, objects ):
	sio.savemat(path, objects)

def readFromMatlabFile( path ):
	return sio.loadmat(path)

if __name__ == '__main__':

	pass