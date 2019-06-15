import numpy as np
import cv2       # import the necessary packages

class Region_Based:

	def __init__(self, bins):                   # store the number of bins for the 3D histogram
		self.bins = bins

	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)       # convert the image to the HSV color space and initialize
		features = []                                        # the features used to quantify the image

		(h, w) = image.shape[:2]                   # grab the dimensions
		(cX, cY) = (int(w * 0.5), int(h * 0.5))    # computing the center of the image


		# construct an elliptical mask representing the center of the image
		#(axesX, axesY) = (int(w) / 2, int(h) / 2)
		#ellipMask = np.zeros(image.shape[:2], dtype = "uint8")  # defining a blank image (filled with 0s to represent a black background)
		#cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		(axesX, axesY) = (int(w) , int(h))
		rectMask = np.zeros(image.shape[:2], dtype = "uint8")  # defining a blank image (filled with 0s to represent a black background)
		cv2.rectangle(rectMask,(0,0),(axesX, axesY),(255,0,0), -1)


		#hist = self.histogram(image, ellipMask)      # extract a color histogram from the elliptical region

		hist = self.histogram(image, rectMask)      # extract a color histgram from the rectangular image

		features.extend(hist)          # update the feature vector

		return features		           # return the feature vector

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist,hist).flatten()       # normalize the histogram

		return hist		 # return the histogram
