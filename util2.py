import os, sys, math, itertools, pickle, movie_genre 
from random import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

num_path = "feature_vectors" #path for  pickle files that store feature vecs of movies
percent_path = "percentage_feature_vectors"
trainingRatio = .80 # divides up the dataset into training/ test sets in a 80-20 ratio. 

def getMovieDataset():
	"""
	Gathers all the names of the movies in the dataset by going through each pickle
	file that stores it's feature vectors. Returns a list of shuffled movie names.
	"""
	if os.path.exists(percent_path):
		filenames = os.listdir(percent_path)
	else: 
		print "Path does not exist: \"" + percent_path + "\""
		return 	
	shuffle(filenames)
	return filenames

def seedTrainSet():
        # What we really want to do here is not to have an equal representation
        # set but to have a large training set with examples from every genre.
        # Our classifier suffers when some genres are more heavily represented
        # in the test set than the training.
        """
        Picks 80% of the most underrepresented genre and then an equivalent number
        of every other genre. Thus, the training set will have equal representation
        of movies.
        """
        genre_dict = movie_genre.get_genre_list()
        min_len = 100
        smallest_genre = "Action"
        for x in genre_dict.keys():
                if len(genre_dict[x]) < min_len:
                        smallest_genre = x
                        min_len = len(genre_dict[x])

        num_per_genre = len(genre_dict[smallest_genre]) * 0.2
        num_per_genre = math.ceil(num_per_genre)

        trainSet = []
        for x in genre_dict.keys():
                files = genre_dict[x]
                shuffle(files)
                for i in range(0, int(num_per_genre)):
                        trainSet.append(files[i])
        return (int(num_per_genre) * len(genre_dict.keys()), trainSet)

# Global variables initiatizing the training and data set. 
dataset = getMovieDataset()
offset, seededTrainSet = seedTrainSet()
numTrainExamples = int(math.ceil(trainingRatio * len(dataset))) - offset
# print (offset)
# print len(seededTrainSet)
dataset = [x for x in dataset if x not in seededTrainSet]
trainFiles = dataset[0:numTrainExamples] + seededTrainSet
testFiles = dataset[numTrainExamples:]

def getColors():
    return pickle.load(open("extra/colors2.p", "rb"))

def getFeatureVectors(dataset, dVec, numbers=False):
	"""
	Returns the feature vectors from the pickle files into NumPy arrays for use with 
	scikit-learn estimators.
	@param list dataset: filenames of the movies whose correct genres are required
	@param dVec DictVectorizer from scikit-learn
        @param bool numbers: If true, then use feature vectors with total number counts.
        If false, use feature vectors with percentage counts. False by default
	"""
        path = num_path if numbers else percent_path
	feature_vectors = []
	for movie in dataset: 		
		feature_vec = pickle.load(open(path + "/" + movie, "rb"))
		feature_vectors.append(feature_vec)
        
	return dVec.transform(feature_vectors)


def getCorrectGenres(dataset):
	""" 
	Returns an array [n_samples, n_classes] corresponding to the genre of each movie 
	that can be fed into a scikit-learn estimator.
	@param list dataset: filenames of the movies whose correct genres are required
	@param mlb MultiLabelBinarizer from scikit-learn
	@return 2-D matrix in which cell [i, j] is 1 if movie i has genre j and 0 otherwise
	"""
	movie_genres = []
	for movie_name in dataset:
		movie_name = os.path.splitext(movie_name)[0].replace('_', ' ')
		#refer to movie_genre.py, which uses the IMDB API to query genres.
		genres = movie_genre.get(movie_name) 
		movie_genres.append(genres) 
	return movie_genres

def printOutput(mlb, testExamples, predicted, correct):
	assert (len(predicted) == len(testExamples)), "No. of samples in predicted and test data don't match"
	assert (len(predicted) == len(correct)), "No. of samples in predicted and correct datasets don't match"
	for i, sample in enumerate(testExamples):
		print sample
		print "PREDICTED: "
		for index, label in enumerate(predicted[i]):
			if label: 
				print "\t" + mlb.classes_[index]
		print "CORRECT: "
		for index, label in enumerate(correct[i]):
			if label: 
				print "\t" + mlb.classes_[index]

def printCorrectness(mlb, testExamples, predicted, correct):
	totCount = 0
	correctCount = 0
	for i, sample in enumerate(predicted):
		for index, label in enumerate(sample):
			if label: 
				totCount +=1 
				if correct[i][index]:
					correctCount +=1 
	correctness = float(correctCount)/totCount
	print "CORRECTNESS: " + str(correctness)

def printAccuracyByGenre(mlb, testExamples, predicted, correct):
	print "ACCURACY BY GENRE:"
	for index, genre in enumerate(mlb.classes_):
		totCount = 0
		correctCount = 0
		for i, sample in enumerate(correct):
			if sample[index]: 
				totCount +=1
				if predicted[i][index]:
					correctCount += 1
		if totCount !=0: 
			accuracy = float(correctCount)/totCount
			print genre + ": " + str(accuracy)
		else: 
			print genre + ": Does not appear in test data" 