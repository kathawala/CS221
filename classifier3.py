# ONE-VS-REST CLASSIFIER
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
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
    def __init__(self, numbers=True):
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
    # In this perhaps pass in customizable parameters so
    # __init__(self, loss="hinge", penalty="l2")
    # This way, we can try out different loss functions easily
    def __init__(self):
        self.trainExamples = ['exodus_gods_and_kings.p', 'how_to_train_your_dragon_2.p', 'bears.p', 'see_no_evil_2.p', 'addicted.p', "the_internet's_own_boy_the_story_of_aaron_swartz.p", 'the_salt_of_the_earth.p', 'the_other_woman.p', 'project_almanac.p', 'edge_of_tomorrow.p', 'maya_the_bee_movie.p', 'cowspiracy_the_sustainability_secret.p', "let's_be_cops.p", "winter's_tale.p", 'the_trip_to_italy.p', 'yellowbird.p', 'alexander_and_the_terrible_horrible_no_good_very_bad_day.p', 'rosewater.p', 'the_hero_of_color_city.p', 'endless_love.p', 'dracula_untold.p', 'dumb_and_dumber_to.p', 'tomorrowland.p', 'the_hunger_games_mockingjay_part_1.p', 'tammy.p', 'hot_tub_time_machine_2.p', 'lucy.p', 'the_lego_movie.p', 'the_judge.p', 'cake.p', 'st_vincent.p', 'black_or_white.p', 'american_sniper.p', 'mr_peabody_&_sherman.p', 'this_is_where_i_leave_you.p', 'x-men_days_of_future_past.p', 'non-stop.p', 'get_on_up.p', 'the_fault_in_our_stars.p', 'song_one.p', 'robocop.p', 'into_the_storm.p', 'a_most_wanted_man.p', 'the_good_lie.p', 'wild.p', 'the_maze_runner.p', 'beyond_the_lights.p', 'divergent.p', 'spring.p', 'as_above_so_below.p', 'noble.p', 'hercules.p', 'i-lived&y=2015.p', 'night_at_the_museum_secret_of_the_tomb.p', 'planes:fire_&_rescue.p', 'old_fashioned.p', 'the_identical.p', 'dawn_of_the_planet_of_the_apes.p', 'cabin_fever_patient_zero.p', 'ride_along.p', 'dear_white_people.p', 'if_i_stay.p', 'red_army.p', 'the_boxtrolls.p', 'captain_america_the_winter_soldier.p', 'virunga.p', 'the_interview.p', 'earth_to_echo.p', 'a_walk_among_the_tombstones.p', 'persecuted.p', 'the_book_of_life.p', 'unbroken.p', 'the_drop.p', 'need_for_speed.p', 'brick_mansions.p', 'maleficent.p', 'blended.p', "devil's_due.p", 'jessabelle.p', 'fear_clinic.p', 'gone_girl.p', 'birdman_or_the_unexpected_virtue_of_ignorance.p', 'kill_the_messenger.p', 'my_little_pony_equestria_girls.p', 'rio_2.p', 'big_hero_6.p', 'guardians_of_the_galaxy.p', 'noah.p', 'the_hobbit_the_battle_of_the_five_armies.p', 'i_frankenstein.p', 'the_november_man.p', 'the_pyramid.p', 'and_so_it_goes.p', 'birdman_or_the_unexpected_virtue_of_ignorance.p', 'inherent_vice.p', 'merchants_of_doubt.p', 'iris.p', 'lambert,_stamp.p']
        self.testExamples = [x for x in util2.getMovieDataset() if x not in self.trainExamples]
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
                'colors': 0.5,
                'subs': 0.5,
                },
                )),
            ('est', KNeighborsClassifier()),
            ])
        # OneVsRestClassifier used for prediction
        self.classif = OneVsRestClassifier(self.pipeline)
                
    def learnPredictor(self, numbers=False):
        train_genres = self.mlb.fit_transform(util2.getCorrectGenres(self.trainExamples))
        self.classif.fit(self.trainExamples, train_genres)
        return train_genres

    def predict(self, numbers=False):
        return self.classif.predict(self.testExamples)