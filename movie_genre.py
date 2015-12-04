#!/bin/env python

import requests
import argparse

IMDB = "http://www.omdbapi.com/"
genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", \
"Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", \
"Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]

def get(title):
    """Gets the movie genre of the specified movie"""
    r = requests.get(IMDB + "?t=" + title)
    return r.json()['Genre']
