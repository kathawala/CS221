import os, sys, math, itertools, pickle, movie_genre 
from random import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

path = "feature_vectors" #path for  pickle files that store feature vecs of movies 
trainingRatio = .80 # divides up the dataset into training/ test sets in a 80-20 ratio. 

def getMovieDataset():
	"""
	Gathers all the names of the movies in the dataset by going through each pickle
	file that stores it's feature vectors. Returns a list of shuffled movie names.
	"""
	if os.path.exists(path):
		filenames = os.listdir(path)
	else: 
		print "Path does not exist: \"" + path + "\""
		return 	
	shuffle(filenames)
	return filenames

# Global variables initiatizing the training and data set. 
dataset = getMovieDataset()
numTrainExamples = int(math.ceil(trainingRatio * len(dataset)))
trainFiles = dataset[0:numTrainExamples]
testFiles = dataset[numTrainExamples:]

def getColors():
    return pickle.load(open("extra/colors2.p", "rb"))

def getFeatureVectors(dataset):	
	"""
	Returns the feature vectors from the pickle files into NumPy arrays for use with 
	scikit-learn estimators.
	@param list dataset: filenames of the movies whose correct genres are required
	"""
	allColors = getColors()
	l = [dict((feature,0) for feature in allColors)]
	v = DictVectorizer()
	v.fit(l)
	feature_vectors = []
	for movie in dataset: 		
		feature_vec = pickle.load(open(path + "/" + movie, "rb"))
		feature_vectors.append(feature_vec)
	return v.transform(feature_vectors)


def getCorrectGenres(dataset):
	""" 
	Returns an array [n_samples, n_classes] corresponding to the genre of each movie 
	that can be fed into a scikit-learn estimator.
	@param list dataset: filenames of the movies whose correct genres are required
	@return 2-D matrix in which cell [i, j] is 1 if movie i has genre j and 0 otherwise
	"""
	movie_genres = []
	for movie_name in dataset:
		movie_name = os.path.splitext(movie_name)[0].replace('_', ' ')
		#refer to movie_genre.py, which uses the IMDB API to query genres.
		genres = set(movie_genre.get(movie_name).split(", ")) 
		movie_genres.append(genres) 
	mlb = MultiLabelBinarizer()
	return mlb.fit_transform(movie_genres)