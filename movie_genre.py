#!/bin/env python

import requests
import argparse

IMDB = "http://www.omdbapi.com/"

def get(title):
    """Gets the movie genre of the specified movie"""
    r = requests.get(IMDB + "?t=" + title)
    return r.json()['Genre']
