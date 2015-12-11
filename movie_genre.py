#!/bin/env python

import requests
import argparse
import os
import pickle

IMDB = "http://www.omdbapi.com/"
genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", \
"Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", \
"Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]
genres_we_dont_care_about = ["War", "Western", "Sport", "Music", "Musical", "Mystery", \
                             "Family", "Film-Noir", "History", "Crime", "Thriller", \
                             "Biography", "Romance", "Fantasy", "Sci-Fi"]

genre_dict = pickle.load(open("genre_dict.p", "rb"))

def get(title):
    """Gets the movie genre of the specified movie"""
    # r = requests.get(IMDB + "?t=" + title + "&y=2014")
    # if "Error" in r.json().keys():
        # r = requests.get(IMDB + "?t=" + title)
    # genres = r.json()['Genre']
    # genres = set(genres.split(", "))
    # genres = set([u'Documentary']) if 'Documentary' in genres else genres.difference(genres_we_dont_care_about)
    genres = []
    title = title.replace(' ', '_') + ".p"
    for x in genre_dict.keys():
        if title in genre_dict[x]:
            genres.append(x)
    genres = set([u'Documentary']) if 'Documentary' in genres else set(genres).difference(genres_we_dont_care_about)
    return genres

def get_genre_list():
    # prev_dir = os.getcwd()
    # os.chdir("feature_vectors")
    # files = os.listdir(os.getcwd())
    # genre_dict = {}
    # for f in files:
    #     f = f.strip('\n')
    #     prev_f = f
    #     title = f[:-2]
    #     title = title.replace('_', ' ')
    #     genres = get(title)
    #     for genre in genres:
    #         if genre in genre_dict:
    #             genre_dict[genre].append(prev_f)
    #         else:
    #             genre_dict[genre] = [prev_f]

    # os.chdir(prev_dir)
    # return genre_dict
    return pickle.load(open("genre_dict.p", "rb"))
    
    
def print_list():
    genre_dict = get_genre_list()
    for x,y in genre_dict.items():
        print (x + "\t\t" + str(len(y)))