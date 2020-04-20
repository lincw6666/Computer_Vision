import os
import cv2
import numpy as np
from convolution import convolution


def Interpolation_x(x, padding = None):
	pad = 0
	if padding:
		pad = 1
	inter_x = np.zeros(x.shape[0] * 2, dtype = np.float32)
	print(inter_x.shape)
	inter_x[0] = x[0]
	inter_x[1] = (x[0] * 2 + x[1] * 1) / 3.0
	inter_x[2] = (x[0] * 1 + x[1] * 2) / 3.0
	for e in range(1, x.shape[0] - 1):
		inter_x[e * 2 + 1] = x[e]
		inter_x[e * 2 + 2] = (x[e] + x[e+1]) /2.0

	inter_x[(x.shape[0] - 1) * 2 +1] = x[x.shape[0] - 1]
	return inter_x.reshape(-1, 1)


def downsampling(img):
	
	return img[0 : img.shape[0] : 2, 0 : img.shape[0] : 2]

def upsampling(img):
	
	up = cv2.pyrUp(img)
	return up.reshape(up.shape[0], up.shape[1], -1)


a = np.array([1,2,3,4,5])#.reshape(-1,1)
print(a)
print(Interpolation_x(a))

x = []

for e in range(3):
	if len(x) == 0:
		x = Interpolation_x(a)
	else:
		x = np.hstack((x, Interpolation_x(a)))
print(x)
imgpath = './task1and2_hybrid_pyramid'
Pyramids_layer = 5
files = os.listdir(imgpath)
#Gaussian Filter
GF_size = 5
x, y = np.mgrid[ -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1, -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1]
gaussian_kernel = np.exp(-( x**2 + y**2) )

#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

for file in files:
	break
	gaussian_img = []
	if file == '.DS_Store':
		continue
	filename, extension = file.split('.')
	img = cv2.imread(imgpath + '/' + file)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	'''
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	img = cv2.resize(img, (512, 512)).reshape(512, 512, -1)
	print(img.shape)
	img = convolution(img, gaussian_kernel)
	for i in range(Pyramids_layer):
		#GF_img = convolution(img, gaussian_kernel)
		gaussian_img.append(img)
		img = downsampling(img)


	for i in range(Pyramids_layer - 1,-1,-1):

		up_error = gaussian_img[i] - upsampling(img)
		print(up_error.shape)
		cv2.imwrite(filename + '_laplacian_layer{}'.format(i) + '.' + extension, up_error)
		img = gaussian_img[i]

	cv2.imshow("error", up_error + img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	cv2.imshow("GF_img", GF_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	#break


| a | (2a + b) /3 | 