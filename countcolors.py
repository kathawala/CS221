from __future__ import division
from PIL import Image
import pickle
import sys
import colorgetter
import locatefiles
import collections

if sys.argv[1]:
    name = sys.argv[1]
    feature_vec = collections.Counter()
    img_files = locatefiles.getFiles(name)
    num_files = len(img_files)
    for f in img_files:
        img = Image.open(f)
        colors = []
        for i, pix in enumerate(img.getdata()):
            actual_name, closest_name = colorgetter.get_colour_name(pix)
            color = actual_name if actual_name != None else closest_name
            if color in colors:
                continue
            colors.append(color)
        feature_vec.update(colors)
            
    pickle.dump(feature_vec, open("/home/cs221/project/feature_vectors/" + name.lower() + ".p", "wb"), 2)
    percentage_vec = { x:y/num_files for x,y in vec.items() }
    pickle.dump(percentage_vec, open("/home/cs221/project/percentage_vectors/" + name.lower() + ".p", "wb"), 2)
else:
    print ("Please specify a video file to count the colors of")
