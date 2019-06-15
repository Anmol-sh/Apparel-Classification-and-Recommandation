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
	# args={'index':"Data/index/women_tops.csv",
	# 	"query":"send.jpg",
	# 	"result_path":"Data/women_tops"}
	cd = Region_Based((8, 12, 3))      #initialize the image descriptor. Here we specify the number of bins for hue, saturation and value.

	# load the query image and describe it
	query = cv2.imread(query_image)
	#grey_query=cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
	features = cd.describe(query)


	searcher = Searcher(indexFile)     
	# perform the search
	results = searcher.search(features)

	# cv2.imshow("Query", query)     # display the query
	# #cv2.imshow("Greyscale version of the query",grey_query)
	# cv2.waitKey(0)
	# loop over the results
	i=0
	print("Got results")
	rmtree(config.resultDir)
	mkdir(config.resultDir)
	for (score, resultID) in results:
		print(catFile + "/" + resultID)     # load the result image and display it
		copy2(catFile + "/" + resultID, config.resultDir)
		# cv2.imshow("Results", result)
		# print(results[i])
		i=i+1
		# cv2.waitKey(0)
