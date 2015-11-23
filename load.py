import pickle
import sys

name = sys.argv[1]

vec = pickle.load(open("feature_vectors/" + name.lower() + ".p", "rb"))
print (vec)
