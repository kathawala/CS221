import os, sys
import movie_genre 

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

def getMovieGenres():
	movie_genres = []
	path = "feature_vectors"
	if os.path.exists(path):
		filenames = os.listdir(path)
	else: 
		print "Path does not exist: \"" + path + "\""
		return 
	for movie_name in filenames:
		movie_name = os.path.splitext(movie_name)[0].replace('_', ' ')
		genres = set(movie_genre.get(movie_name).split(", "))
		movie_genres.append(genres)
	return movie_genres

def getFeatureVectors(trainFile):
	v = DictVectorizer(sparse=False)
	feature_vec = pickle.load(open(feature_vec_path + f, "rb"))
	return v.fit_transform(D)

def getTrainExamples():         
	movie_genres = getMovieGenres()
	mlb = MultiLabelBinarizer()
	#mlb.fit_transform(movie_genres)
	classif = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))
	# replace x with training example shit 
    classif.fit(x, mlb.fit_transform(movie_genres))


getTrainExamples()