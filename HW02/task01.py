import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

os.chdir('your path')

def GaussianFilter(x , y , D0 , highpass = True):
	centerx = int(x/2) + 1 if x % 2 == 1 else int(x/2)
	centery = int(y/2) + 1 if y % 2 == 1 else int(y/2)
	Filter = np.zeros([x,y])
	if highpass == True:
		for i in range(x):
			for j in range(y):
					Filter[i,j] = 1 - (math.exp(-1 * ((i - centerx)**2 + (j - centery)**2) / (2 * D0**2)))
	else:
		for i in range(x):
			for j in range(y):
					Filter[i,j] = math.exp(-1 * ((i - centerx)**2 + (j - centery)**2) / (2 * D0**2))
	return Filter

def ShiftImage(img):
	shifted_img = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			shifted_img[i,j] = shifted_img[i,j] * ((-1)**(i+1+j+1))
	return shifted_img


def Filter(img , mask):
	shifted_img = ShiftImage(img)
	shifted_f = np.fft.fft2(shifted_img)
	filtered_f = shifted_f * mask
	return ShiftImage(np.fft.ifft2(filtered_f))

def Highpass(img , D0):
	x , y = img.shape
	return Filter(img , GaussianFilter(x , y , D0 , highpass = True))

def Lowpass(img , D0):
	x , y = img.shape
	return Filter(img , GaussianFilter(x , y , D0 , highpass = False))

def hybridImage(img1, img2, img1D0, img2D0):
	highPassed = Highpass(img1, img1D0)
	lowPassed = Lowpass(img2, img2D0)
	return highPassed + lowPassed

##############################################
# For color_image
def Filter_Color(img , mask):
	shifted_img0 = ShiftImage(img[:,:,0])
	shifted_img1 = ShiftImage(img[:,:,1])
	shifted_img2 = ShiftImage(img[:,:,2])
	shifted_f0 = np.fft.fft2(shifted_img0)
	shifted_f1 = np.fft.fft2(shifted_img1)
	shifted_f2 = np.fft.fft2(shifted_img2)
	filtered_f0 = shifted_f0 * mask
	filtered_f1 = shifted_f1 * mask
	filtered_f2 = shifted_f2 * mask
	filter_back = np.zeros(img.shape , dtype = complex)
	filter_back[:,:,0] = ShiftImage(np.fft.ifft2(filtered_f0))
	filter_back[:,:,1] = ShiftImage(np.fft.ifft2(filtered_f1))
	filter_back[:,:,2] = ShiftImage(np.fft.ifft2(filtered_f2))
	return filter_back

def Highpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , GaussianFilter(x , y , D0 , highpass = True))

def Lowpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , GaussianFilter(x , y , D0 , highpass = False))

def hybridImage_Color(img1, img2, img1D0, img2D0):
	highPassed = Highpass_Color(img1, img1D0)
	lowPassed = Lowpass_Color(img2, img2D0)
	return highPassed + lowPassed

img1 = cv2.imread("4_einstein.bmp" )
img2 = cv2.imread("4_marilyn.bmp" )

Highpass_image = Highpass_Color(img1 , 20)
Lowpass_image = Lowpass_Color(img2 , 20)

hybrid = hybridImage_Color(img1, img2, 25, 10)

cv2.imshow('hybrid' , hybrid.real.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
