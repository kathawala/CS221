import pickle
import util
import math
import sys

# IMPORTANT NOTE
# We should use feature extraction to determine which colors are important
# to detect and what features about the file (runtime, size, etc.) are relevant

# |trainExamples| and |testExamples| are both lists of (x,y) tuples
# where x is a feature vector in the form of a counter, whose keys are
# color names and whose values are the number of stills in which that color
# was seen. y is the score of the example (+1 means that the director is Anderson
# and -1 means the director is not Anderson)
# Thus, trainExamples[0][1] will print out the score of the first example
class predictor():
    def __init__(self, genre):
        self.genre = genre
        self.trainExamples = util.getTrainExamples(self.genre)
        self.testExamples = util.getTestExamples(self.genre)
        self.weights = {x : 0 for x in util.getColors()}

    def predict(self, x):
        print "Learned Score:" + str(util.dotProduct(self.weights, x))
        return math.copysign(1.0, util.dotProduct(self.weights, x))

    def learn(self, trainExamples):
        numIters = 10
        step = 0.0001
        for i in range(numIters):
            for feature_vec, y in trainExamples:
                score = util.dotProduct(self.weights, feature_vec)
                dloss = {}
                if score*y > 1:
                    continue
                else:
                    util.increment(dloss, -y, feature_vec)
                util.increment(self.weights, -step, dloss)

genre = sys.argv[1]
p = predictor(genre)

if p.trainExamples is None or p.testExamples is None:
    print "Please specify a valid genre"
    exit(1)
    
p.learn(p.trainExamples)
for ex in p.testExamples:
    print "Corect:" + str(ex[1])
    print "Predicted: " +str((p.predict(ex[0])))
