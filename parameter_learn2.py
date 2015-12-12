# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

######################################################
# PRUNING HYPERPARAMETERS FOR ONE VS REST CLASSIFIER #
######################################################

from __future__ import print_function

from pprint import pprint
from time import time
import logging
from scipy.stats import randint as sp_randint

import numpy as np
import pickle

import util2, classifier2, movie_genre, os, sys

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'union__use_idf': (True, False),
    #'union__norm': ('l1', 'l2'),
    'estimator__union__transformer_weights':({'colors': 0.35,'subs': 0.65}, {'colors': 0.5,'subs': 0.5},
                                            {'colors': 0.2,'subs': 0.8}),
    'estimator__union__subs__tfidf__max_features': (3, 5, 10, 15),
    'estimator__sgd__alpha': (0.001, 0.0001, 0.00001, 0.000001),
    'estimator__sgd__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
    'estimator__sgd__penalty': ('l2', 'elasticnet'),
    'estimator__sgd__n_iter': (10, 20, 50, 80, 100, 150),
}

# parameters = {
#     #'vect__max_df': (0.5, 0.75, 1.0),
#     #'vect__max_features': (None, 5000, 10000, 50000),
#     # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     #'union__use_idf': (True, False),
#     #'union__norm': ('l1', 'l2'),
#     'estimator__union__transformer_weights':({'colors': 0.35,'subs': 0.65}, {'colors': 0.5,'subs': 0.5},
#                                             {'colors': 0.2,'subs': 0.8}),
#     'estimator__union__subs__tfidf__max_features': sp_randint(3, 10),
#     'estimator__sgd__alpha': sp_randint(0.00001, 0.000001),
#     'estimator__sgd__loss': ('hinge', 'modified_huber', 'squared_hinge', 'perceptron'),
#     'estimator__sgd__penalty': ('l2', 'elasticnet'),
#     'estimator__sgd__n_iter': sp_randint(10, 80),
# }

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    for i in range(0, 20):
      p = classifier2.predictor()
      print (p.trainExamples)
      grid_search = RandomizedSearchCV(p.classif, parameters, n_jobs=-1, verbose=1, error_score=0)

      print("Performing grid search...")
      print("parameters:")
      pprint(parameters)
      t0 = time()
      grid_search.fit(p.trainExamples, p.mlb.fit_transform(util2.getCorrectGenres(p.trainExamples)))
      print("done in %0.3fs" % (time() - t0))
      print()
      print("Best score: %0.3f" % grid_search.best_score_)
      print("Best parameters set:")
      best_parameters = grid_search.best_estimator_.get_params()
      for param_name in sorted(parameters.keys()):
          print("\t%s: %r" % (param_name, best_parameters[param_name]))
      print ("###########################################################################")
