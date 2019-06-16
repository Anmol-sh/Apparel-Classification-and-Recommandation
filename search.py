from region_based import Region_Based
from searcher import Searcher
import argparse
import cv2
from shutil import copy2, rmtree
from os import mkdir
import config
from prediction import predict

def search_sample():
	query_image	= config.queryImage
	category = predict(query_image)
	indexFile = config.indexDir + '/' + category + '.csv'
	catFile = config.dataFile + '/' + category
	
	cd = Region_Based((8, 12, 3))      #initialize the image descriptor. Here we specify the number of bins for hue, saturation and value.

	# load the query image and describe it
	query = cv2.imread(query_image)
	features = cd.describe(query)


	searcher = Searcher(indexFile)     
	# perform the search
	results = searcher.search(features)

	i=0
	print("Got results")
	rmtree(config.resultDir)
	mkdir(config.resultDir)
	for (score, resultID) in results:
		print(catFile + "/" + resultID)     # load the result image and display it
		copy2(catFile + "/" + resultID, config.resultDir)
		i=i+1

if __name__=='__main__':
    search_sample()