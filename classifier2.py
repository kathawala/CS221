# ONE-VS-REST CLASSIFIER

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
    # In this perhaps pass in customizable parameters so
    # __init__(self, loss="hinge", penalty="l2")
    # This way, we can try out different loss functions easily
    def __init__(self):
        self.trainExamples = ['tomorrowland.p', 'the_trip_to_italy.p', 'john_wick.p', 'the_grand_budapest_hotel.p', 'iris.p', 'the_book_of_life.p', 'earth_to_echo.p', 'and_so_it_goes.p', 'endless_love.p', 'unbroken.p', 'tammy.p', 'the_interview.p', 'blended.p', 'fear_clinic.p', 'divergent.p', 'horrible_bosses_2.p', 'as_above_so_below.p', 'merchants_of_doubt.p', 'ride_along.p', 'my_little_pony_equestria_girls.p', 'a_most_wanted_man.p', 'red_army.p', 'rosewater.p', 'the_judge.p', 'dawn_of_the_planet_of_the_apes.p', 'x-men_days_of_future_past.p', 'the_hunger_games_mockingjay_part_1.p', 'noble.p', 'paranormal_activity_the_marked_ones.p', "winter's_tale.p", 'veronica_mars.p', 'exodus_gods_and_kings.p', 'spring.p', 'the_hobbit_the_battle_of_the_five_armies.p', 'guardians_of_the_galaxy.p', 'captain_america_the_winter_soldier.p', 'planes:fire_&_rescue.p', 'the_fault_in_our_stars.p', 'muppets_most_wanted.p', "let's_be_cops.p", 'the_good_lie.p', 'addicted.p', 'cowspiracy_the_sustainability_secret.p', 'the_raid_2.p', 'kill_the_messenger.p', 'teenage_mutant_ninja_turtles.p', 'the_drop.p', 'birdman_or_the_unexpected_virtue_of_ignorance.p', 'the_maze_runner.p', 'brick_mansions.p', 'interstellar.p', 'noah.p', 'cabin_fever_patient_zero.p', 'st_vincent.p', 'night_at_the_museum_secret_of_the_tomb.p', 'transcendence.p', 'american_sniper.p', 'old_fashioned.p', 'jessabelle.p', 'home&y=2015.p', 'wild.p', 'inherent_vice.p', 'hercules.p', 'project_almanac.p', 'need_for_speed.p', 'the_pyramid.p', 'the_lego_movie.p', 'this_is_where_i_leave_you.p', 'maleficent.p', 'into_the_storm.p', 'yellowbird.p', 'dumb_and_dumber_to.p', 'hot_tub_time_machine_2.p', 'robocop.p', 'the_other_woman.p', 'the_purge_anarchy.p', 'black_or_white.p', 'the_town_that_dreaded_sundown.p', 'i-lived&y=2015.p', 'annabelle.p', 'jersey_boys.p', 'the_theory_of_everything.p', 'dracula_untold.p', 'the_hero_of_color_city.p', 'the_nut_job.p', 'how_to_train_your_dragon_2.p', 'maya_the_bee_movie.p', 'edge_of_tomorrow.p', 'rio_2.p', 'i_frankenstein.p', 'lucy.p', 'big_hero_6.p', 'mr_peabody_&_sherman.p', 'vampire_academy.p', 'the_boxtrolls.p', "the_internet's_own_boy_the_story_of_aaron_swartz.p", 'the_salt_of_the_earth.p', 'citizenfour.p']
        self.testExamples = [x for x in util2.getMovieDataset() if x not in self.trainExamples]
        print self.testExamples
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
                    ('tfidf', TfidfVectorizer(strip_accents='ascii', max_features=3)),
                    ])),
                ],
                transformer_weights={
                'colors': 0.5,
                'subs': 0.5,
                },
                )),
            ('sgd', SGDClassifier(alpha= 1e-06, loss="modified_huber", n_iter= 80, penalty="l2")),
            ])
		# OneVsRestClassifier used for prediction
        self.classif = OneVsRestClassifier(self.pipeline)
                
    def learnPredictor(self, numbers=False):
        train_genres = self.mlb.fit_transform(util2.getCorrectGenres(self.trainExamples))
        self.classif.fit(self.trainExamples, train_genres)
        return train_genres

    def predict(self, numbers=False): 
		return self.classif.predict(self.testExamples)