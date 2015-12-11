# NEARESET NEIGHBOURS

from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pickle

import util2, movie_genre, os, sys

class Movie_Data_Aggregator(BaseEstimator, TransformerMixin):
    """Creates a dict where the key is the name of a feature and
    second index is the samples of that feature. For example:

    data = {'colors': [unrboken_color_vector, wild_color_vector, etc]
            'subs': [unbroken.txt, wild.txt, etc]}
    """
    def __init__(self, numbers=False):
        self.path = "feature_vectors" if numbers else "percentage_feature_vectors"
        self.sub_path = "subtitles"
            
    def fit(self, x, y=None):
        return self

    def transform(self, files):
        features = np.recarray(shape=(len(files), ),
                               dtype=[('colors', object), ('subs', object)])
        for i,f in enumerate(files):
            feature_vec = pickle.load(open(self.path + "/" + f, "rb"))
            features['colors'][i] = feature_vec
            sub_file = f.replace(".p", ".txt")
            features['subs'][i] = self.sub_path + "/" + sub_file
        return features      
            
class Data_Selector(BaseEstimator, TransformerMixin):
    """Grabs specified data from movie_data object (subtitles, color vector, etc)"""
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class predictor():
    def __init__(self):
        self.trainExamples = util2.trainFiles
        self.testExamples = util2.testFiles
        # Standard DictVectorizer fitted with all colors as the features. 
        self.dVec = DictVectorizer(sparse=False)
        self.dVec.fit([dict((feature,0) for feature in util2.getColors())])
        # Standard MultiLabelBinarizer with all genre names 
        self.mlb = MultiLabelBinarizer()
        self.pipeline = Pipeline([
            ('organizeData', Movie_Data_Aggregator()),
            ('union', FeatureUnion(
                transformer_list = [
                ('colors', Pipeline([
                    ('selector', Data_Selector(key='colors')),
                    ('dVec', self.dVec),
                    ])),
                ('subs', Pipeline([
                    ('selector', Data_Selector(key='subs')),
                    ('tfidf', TfidfVectorizer(strip_accents='ascii', max_features=15)),
                    ])),
                ],
                transformer_weights={
                'colors': 0.35,
                'subs': 0.65,
                },
                )),
            ])
        # OneVsRestClassifier used for prediction
        self.classif = RadiusNeighborsClassifier(self.pipeline)
                
    def learnPredictor(self, numbers=False):
        train_genres = self.mlb.fit_transform(util2.getCorrectGenres(self.trainExamples))
        self.classif.fit(self.trainExamples, train_genres)
        return train_genres

    def predict(self, numbers=False): 
		return self.classif.predict(self.testExamples)