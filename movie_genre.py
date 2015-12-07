#!/bin/env python

import requests
import argparse
import os

IMDB = "http://www.omdbapi.com/"
genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", \
"Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", \
"Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]
genres_we_dont_care_about = ["War", "Western", "Sport", "Music", "Musical", "Mystery", \
                             "Family", "Film-Noir", "History", "Crime", "Thriller", \
                             "Biography", "Sci-Fi", "Romance", "Fantasy"]

def get(title):
    """Gets the movie genre of the specified movie"""
    r = requests.get(IMDB + "?t=" + title + "&y=2014")
    if "Error" in r.json().keys():
        r = requests.get(IMDB + "?t=" + title)
    genres = r.json()['Genre']
    genres = set(genres.split(", "))
    genres = genres.difference(genres_we_dont_care_about)
    return genres

def print_list():
    with open("movies.txt") as f:
        content = [x.strip('\n').replace('_', ' ').replace('\\', '') for x in f.readlines()]
        genre_dict = { x:get(x) for x in content}

    for x,y in genre_dict.items():
        print (x + "\t\t" + repr(y))