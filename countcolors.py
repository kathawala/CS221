from PIL import Image
import pickle
import sys
import colorgetter
import locatefiles
import collections

if sys.argv[1]:
    name = sys.argv[1]
    feature_vec = collections.Counter()
    for f in locatefiles.getFiles(name):
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
else:
    print ("Please specify a video file to count the colors of")
