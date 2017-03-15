from myio import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from metrics import getKx2CVScores

from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators

import numpy as np

labels = getLabelsFromCSVFile( 'dataset/train_answers.csv' )
[X_, y] = getDatasetFromCSVFile( 'dataset/train.csv', labels )
X = X_[:,4:]

clf = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.1) # LogisticRegression( C = 1000 )

def eval_func(chromosome):

	features = []

	for i in xrange(len(chromosome)):
		if chromosome[i] == 1:
			features.append(i)

	scores = getKx2CVScores(clf, X[:,features], y, k = 1)

   	return np.mean(scores[0])

genome = G1DBinaryString.G1DBinaryString(X.shape[1])

genome.evaluator.set(eval_func)
genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GTournamentSelector)

ga.setElitism(True)
ga.setGenerations(20)
ga.setPopulationSize(25)
ga.setMultiProcessing(True)

ga.evolve(freq_stats=1)
best = ga.bestIndividual()

print best

np.save('nGB', best)