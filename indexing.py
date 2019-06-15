from region_based import Region_Based
import argparse
import glob
import cv2  
from os import listdir
from os.path import isfile, join
import config

def index_files():
    data_path = config.dataFile
    cat = [f for f in listdir(data_path) if not isfile(join(data_path, f))]

    for c in cat:
        cd = Region_Based((8, 12, 3)) # initialize the color descriptor
        index_path=config.indexDir + '/' + c
        index_path=index_path+'.csv'
        output = open(index_path, "w")         # open the output index file for writing
        # use glob to grab the image paths and loop over them
        data_path=data_path + '/' + c + "/"
        for imagePath in glob.glob(data_path + "/*.jpg"):
            # extract the image ID (i.e. the unique filename) from the image
            # path and load the image itself
            imageID = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            features = cd.describe(image)       # describe the image

            # write the features to file
            features = [str(f) for f in features]
            output.write("%s,%s\n" % (imageID, ",".join(features)))

        output.close()         # close the index file

