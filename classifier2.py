from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import util2, movie_genre, os, sys

class predictor():
	def __init__(self):
		self.trainExamples = util2.trainFiles
		self.testExamples = util2.testFiles
		self.classif = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))

	def learnPredictor(self):   
		self.classif.fit(util2.getFeatureVectors(self.trainExamples), util2.getCorrectGenres(self.trainExamples))

	def predict(self): 
		print self.classif.classes_
		print self.classif.predict(util2.getFeatureVectors(self.testExamples))

p = predictor()
p.learnPredictor()
p.predict()

for movie in p.testExamples: 
	print movie
	movie_name = os.path.splitext(movie)[0].replace('_', ' ')
		#refer to movie_genre.py, which uses the IMDB API to query genres.
	genres = set(movie_genre.get(movie_name).split(", ")) 
	print movie_genre.get(movie_name)