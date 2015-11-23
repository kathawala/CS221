import glob, os
import sys

def getFiles(movie_name):
    os.chdir("rgb")
    local_files = glob.glob(movie_name + "*.bmp")
    abs_files = [os.path.abspath(x) for x in local_files]
    return abs_files
