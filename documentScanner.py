#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:35:46 2020

@author: prarthanabhat
"""
import cv2
import math
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import distance as dist

# Can add for different kind of blurring : median
def doBlurring(image, intensity = 3):
    return cv2.medianBlur(image, intensity)

def doDilate(image, ksize = (3,3), nround = 2):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
    # Apply dilate function on the input image
    return cv2.dilate(image, kernel1, iterations= nround)

# Can add for different kind of blurring : median
def doThresh(image, low=50, high=200):
    return cv2.threshold(imageDilated, low, high, cv2.THRESH_BINARY)[1]

def orderPoints(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

img = cv2.imread('scanned-form.jpg')

# Median blur
imgBlur = doBlurring(img, 5)
plt.imshow(imgBlur[:,:,::-1])

# To Gray
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
plt.imshow(imgGray, cmap = 'gray')

# Dilate
imageDilated = doDilate(imgGray, (3,3), 5)
plt.imshow(imageDilated, cmap = 'gray');plt.title("Dilated Image");

# Thresholding
thresh = doThresh(imageDilated, 200, 255)
plt.imshow(thresh,cmap='gray')

# Find all contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))
print("\nHierarchy : \n{}".format(hierarchy))

# Get maximum area of countour
areaFlag = 0
n = 0
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > areaFlag:
        areaFlag = cv2.contourArea(contours[i])
        n = i
cv2.contourArea(contours[n])

# Contour with Max Area
image = img.copy()
cv2.drawContours(image, contours[n], -1, (0,0,255), 10);
plt.imshow(image[:,:,::-1])

# Get extreme coordinates 
c = contours[n]
extLeft = tuple(c[c[:, :, 0].argmin()][0]) # smallest x-coordinates
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0]) # smallest y-coordinates
extBot = tuple(c[c[:, :, 1].argmax()][0]) #largest y
# Draw
cv2.circle(image, extLeft, 20, (0,255,0), -1);
cv2.circle(image, extRight, 20, (0,255,0), -1);
cv2.circle(image, extTop, 20, (0,255,0), -1);
cv2.circle(image, extBot, 20, (0,255,0), -1);
plt.imshow(image[:,:,::-1])

# Order coordinates in clockwise
pts_src = []
pts_src = np.array([extLeft, extTop, extBot, extRight], dtype=float)
srcPtsOrd = orderPoints(pts_src)

pts_dst = np.array([(5,5), (img.shape[1]-5,5), (img.shape[1]-5,img.shape[0]-5)
                    , (5,img.shape[0]-5)], dtype=float)
# Calculate Homography
h, status = cv2.findHomography(srcPtsOrd, pts_dst)

# Warp source image to destination based on homography
imgTemplate = np.ones(img.shape) * 255
im_out = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))
plt.imshow(im_out[:,:,::-1])


