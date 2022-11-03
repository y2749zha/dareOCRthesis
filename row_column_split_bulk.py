# USAGE
# python row_column_split_bulk.py -f folder_name

import cv2
import numpy as np
import argparse
import os
import glob
from natsort import natsorted

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", type=str,
	help="path to input folder")
args = vars(ap.parse_args())


def split(img, pa):
	# greyscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# threshold
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 
	thresh = 255 - thresh

	# # apply morphology open (optional)
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	# morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# morph = 255 - morph

	# find and draw the upper and lower boundary of each lines
	hist = cv2.reduce(thresh,1, cv2.REDUCE_AVG).reshape(-1)

	th = 2
	H,W = img.shape[:2]
	uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
	lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

	# find and draw the left and right boundary of each lines
	hist_v = cv2.reduce(thresh,0, cv2.REDUCE_AVG).reshape(-1)

	th = 2
	H,W = img.shape[:2]
	left = [x for x in range(W-1) if hist_v[x]<=th and hist_v[x+1]>th]
	right = [x for x in range(W-1) if hist_v[x]>th and hist_v[x+1]<=th]

	thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

	# single line split
	if len(lowers) > len(uppers):
		uppers.append(H)
	elif len(lowers) < len(uppers):
		lowers.insert(0,0)
	else:
		uppers.append(H)
		lowers.insert(0,0)

	lines = ( np.array(uppers) + np.array(lowers) ) // 2.0
	lines = lines.astype(int)

	n = range(len(lines)-1)
	for i in n:
	    if lines[i+1] - lines[i] < 20:
	        lines[i+1] = (lines[i+1] + lines[i]) / 2
	        lines[i] = 0
	        
	lines = [i for i in lines if i != 0]

	for y in lines:
	    cv2.line(thresh, (0,y), (W, y), (255,0,0), 1)

	# single line split
	left = [i for i in left if i != 0]
	right = [i for i in right if i != 0]

	if len(left) == len(right):
		right.insert(0,0)
		left.append(W)		
	elif len(left) < len(right):
		left.append(W)
	else:
		right.insert(0,0)
		
	lines_v = ( np.array(left) + np.array(right) ) // 2.0
	lines_v = lines_v.astype(int)

	n = range(len(lines_v)-1)
	for i in n:
	    if lines_v[i+1] - lines_v[i] < 10:
	        lines_v[i+1] = (lines_v[i+1] + lines_v[i]) / 2
	        lines_v[i] = 0
	
	lines_v = [i for i in lines_v if i != 0]

	if lines_v[0] > 15:
		lines_v.insert(0,0)

	# make sure character don't get cut in the middle (optional)
	n = range(len(lines_v)-1)
	for i in n:
	    if lines_v[i+1] - lines_v[i] < 65:
	        lines_v[i+1] = lines_v[i]
	
	temp = []
	for col_line in lines_v:
		if col_line not in temp:
			temp.append(col_line)
	lines_v = temp


	for x in lines_v:
	    cv2.line(thresh, (x,0), (x, H), (255,0,0), 1)

	# split the file name from extension
	filename = os.path.splitext(pa)

	# crop image
	row = 1
	n = range(len(lines)-1)
	for i in n:
		y = lines[i]
		H = lines[i+1]
		col = 1

		# crop image
		n_v = range(len(lines_v)-1)
		for i_v in n_v:		
			x = lines_v[i_v]
			W = lines_v[i_v+1]
			crop = img[y:H, x:W]

			# crop file with file name
			cv2.imwrite("Crop_150/"+str(filename[0])+"_crop_"+str(row)+"_"+str(col)+".jpg", crop)
			col = col + 1
		row = row + 1

	cv2.imshow("result.png", thresh)
	cv2.waitKey(0)	
	
# read all images from a folder
images = [cv2.imread(image) for image in natsorted(glob.glob(args["folder"] + "/*.jpg"))]
names = [name for name in natsorted(glob.glob(args["folder"] + "/*.jpg"))]

# run the split on all image in the folder
for num in range(len(images)):
	filename = os.path.splitext(names[num])
	pa = str(filename[0]).split("\\")[-1]
	split(images[num], pa)
