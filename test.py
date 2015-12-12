from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
import numpy as np
import pickle

import classifier2, classifier3, util2, movie_genre, os, sys

nlrloss = []
nlraprecision = []
plrloss = []
plraprecision = []

nscoreone = 0
nscoretwo = 0
for i in range(0,10):
	p = classifier2.predictor()
	p.learnPredictor()
	n_predicted = p.predict()
	correct = p.mlb.transform(util2.getCorrectGenres(p.testExamples))

	ny_score = np.array(n_predicted)
	y_true = np.array(correct)
	nscoreone += label_ranking_loss(y_true, ny_score)
	nscoretwo += label_ranking_average_precision_score(y_true, ny_score)

print "LABEL RANKING LOSS: " + str(float(nscoreone)/10)
print "LABEL RANKING AVERAGE PRECISION: " + str(float(nscoretwo)/10)
# util2.printCorrectness(p.mlb, p.testExamples, n_predicted, correct)
# util2.printAccuracyByGenre(p.mlb, p.testExamples, n_predicted, correct)
# util2.printOutput(p.mlb, p.testExamples, n_predicted, correct)




# print "=========="
# print "PERCENT RESULTS"
# print "LABEL RANKING LOSS:"
# pscoreone = label_ranking_loss(y_true, py_score)
# print (label_ranking_loss(y_true, py_score))
# print "LABEL RANKING AVERAGE PRECISION"
# pscoretwo = label_ranking_average_precision_score(y_true, py_score)
# print (label_ranking_average_precision_score(y_true, py_score))
# nlrloss.append(nscoreone)
# nlraprecision.append(nscoretwo)
# plrloss.append(pscoreone)
# plraprecision.append(pscoretwo)

# print "================"
# print "NUMBERS AVERAGES"
# print "LABEL RANKING LOSS:"
# print (sum(nlrloss) / len(nlrloss))
# print "LABEL RANKING AVERAGE PRECISION"
# print (sum(nlraprecision) / len(nlraprecision))
# print "================"
# print "PERCENT AVERAGES"
# print "LABEL RANKING LOSS:"
# print (sum(plrloss) / len(plrloss))
# print "LABEL RANKING AVERAGE PRECISION"
# print (sum(plraprecision) / len(plraprecision))