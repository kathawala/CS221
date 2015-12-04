from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

import util2, movie_genre, os, sys

class predictor():
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

	def learnPredictor(self):   
		train_feature_vecs = util2.getFeatureVectors(self.trainExamples, self.dVec)
		train_genres = self.mlb.fit_transform(util2.getCorrectGenres(self.trainExamples))
		self.classif.fit(train_feature_vecs, train_genres)

	def predict(self): 
		test_feature_vecs = util2.getFeatureVectors(self.testExamples, self.dVec)
		return self.classif.predict(test_feature_vecs)

p = predictor()
p.learnPredictor()
predicted = p.predict()
correct = p.mlb.transform(util2.getCorrectGenres(p.testExamples))
print "PREDICTED: "
util2.printOutput(p.mlb, p.testExamples, predicted)
print "CORRECT: "
util2.printOutput(p.mlb,p.testExamples, correct)
