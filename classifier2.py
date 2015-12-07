from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
import numpy as np

import util2, movie_genre, os, sys

class predictor():
        # In this perhaps pass in customizable parameters so
        # __init__(self, loss="hinge", penalty="l2")
        # This way, we can try out different loss functions easily
	def __init__(self):
		self.trainExamples = util2.trainFiles
		self.testExamples = util2.testFiles
		# Standard DictVectorizer fitted with all colors as the features. 
		self.dVec = DictVectorizer(sparse=False)
		self.dVec.fit([dict((feature,0) for feature in util2.getColors())])
		# Standard MultiLabelBinarizer with all genre names 
		self.mlb = MultiLabelBinarizer()
		# OneVsRestClassifier used for prediction
		self.classif = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))

	def learnPredictor(self, numbers=True):
		train_feature_vecs = util2.getFeatureVectors(self.trainExamples, self.dVec, numbers)
		train_genres = self.mlb.fit_transform(util2.getCorrectGenres(self.trainExamples))
		self.classif.fit(train_feature_vecs, train_genres)

	def predict(self, numbers=True): 
		test_feature_vecs = util2.getFeatureVectors(self.testExamples, self.dVec, numbers)
		return self.classif.predict(test_feature_vecs)

p = predictor()
p.learnPredictor()
n_predicted = p.predict()
correct = p.mlb.transform(util2.getCorrectGenres(p.testExamples))
# print "PREDICTED: "
# util2.printOutput(p.mlb, p.testExamples, predicted)
# print "CORRECT: "
# util2.printOutput(p.mlb, p.testExamples, correct)
ny_score = np.array(n_predicted)
y_true = np.array(correct)
p.learnPredictor(numbers=False)
p_predicted = p.predict(numbers=False)
py_score = np.array(p_predicted)
print "NUMBERS RESULTS"
print "LABEL RANKING LOSS:"
print (label_ranking_loss(y_true, ny_score))
print "LABEL RANKING AVERAGE PRECISION"
print (label_ranking_average_precision_score(y_true, ny_score))
print "=========="
print "PERCENT RESULTS"
print "LABEL RANKING LOSS:"
print (label_ranking_loss(y_true, py_score))
print "LABEL RANKING AVERAGE PRECISION"
print (label_ranking_average_precision_score(y_true, py_score))