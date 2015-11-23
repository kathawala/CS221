import pickle

feature_vec_path='feature_vectors/'

comedy = ["moonrise2.p", "grandbudapest2.p", "royal2.p"]
animated = ['princess2.p', 'totoro2.p', 'corpse2.p', 'spirited2.p']
drama = ['sweeney2.p', 'chevalier2.p', 'dark2.p']

genre_lists={
    'comedy' : comedy,
    'animated' : animated,
    'drama' : drama
}

trainFiles = animated[:2] + drama[:2] + comedy[:2]
testFiles = animated[2:] + drama[2:] + comedy[2:]

def getTrainExamples(genre):
    trainExamples = []
    for f in trainFiles:
        feature_vec = pickle.load(open(feature_vec_path + f, "rb"))
        if genre:
            score = 1 if f in genre_lists[genre] else -1
            trainExamples.append((feature_vec, score))
        else:
            return None
    return trainExamples

def getTestExamples(genre):
    testExamples = []
    for g in testFiles:
        feature_vec = pickle.load(open(feature_vec_path + g, "rb"))
        if genre:
            score = 1 if g in genre_lists[genre] else -1
            testExamples.append((feature_vec, score))
        else:
            return None
    return testExamples

def getColors():
    return pickle.load(open(feature_vec_path + "colors2.p", "rb"))

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    d1_fl = {key : float(value) for key,value in d1.iteritems()}
    d2_fl = {key : float(value) for key,value in d2.iteritems()}
    if len(d1_fl) < len(d2_fl):
        return dotProduct(d2_fl, d1_fl)
    else:
        return sum(d1_fl.get(f, 0) * v for f, v in d2_fl.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    d1_fl = {key : float(value) for key,value in d1.iteritems()}
    d2_fl = {key : float(value) for key,value in d2.iteritems()}
    for f, v in d2_fl.items():
        d1[f] = d1_fl.get(f, 0) + v * scale
