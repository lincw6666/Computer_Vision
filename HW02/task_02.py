import os
import cv2
import numpy as np
from convolution import convolution

imgpath = './task1and2_hybrid_pyramid'

files = os.listdir(imgpath)
#Gaussian Filter
GF_size = 5
x, y = np.mgrid[ -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1, -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1]
gaussian_kernel = np.exp(-( x**2 + y**2) )

#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

for file in files:
	if file == '.DS_Store':
		continue


	img = cv2.imread(imgpath + '/' + file)
	GF_img = convolution(img, gaussian_kernel)
	'''
	cv2.imshow("GF_img", GF_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	break
