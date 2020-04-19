import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

os.chdir('your path')

def GaussianFilter(x , y , D0 , highpass = True):
	centerx = int(x/2) + 1 if x % 2 == 1 else int(x/2)
	centery = int(y/2) + 1 if y % 2 == 1 else int(y/2)
	if highpass == True:
		return np.array([
			[
				1 - math.exp(-((i-centerx)**2 + (j-centery)**2) / (2 * D0**2))\
					for j in range(y)
			] for i in range(x)
		])
	else:
		return np.array([
			[
				math.exp(-((i-centerx)**2 + (j-centery)**2) / (2 * D0**2))\
					for j in range(y)
			] for i in range(x)
		])

def IdealFilter(x , y , D0 , highpass = True):
	centerx = int(x/2) + 1 if x % 2 == 1 else int(x/2)
	centery = int(y/2) + 1 if y % 2 == 1 else int(y/2)
	if highpass == True:
		return np.array([
			[
				0 if ((i-centerx)**2 + (j-centery)**2)**0.5 < D0 else 1\
					for j in range(y)
			] for i in range(x)
		])
	else:
		return np.array([
			[
				0 if ((i-centerx)**2 + (j-centery)**2)**0.5 > D0 else 1\
					for j in range(y)
			] for i in range(x)
		])

def ShiftImage(img):
	shifted_img = img.copy()
	shifted_img[::2, 1::2] = -img[::2, 1::2]
	shifted_img[1::2, ::2] = -img[1::2, ::2]
	return shifted_img


def Filter(img , mask):
	shifted_img = ShiftImage(img)
	shifted_f = np.fft.fft2(shifted_img)
	filtered_f = shifted_f * mask
	return ShiftImage(np.fft.ifft2(filtered_f))

def GaussianHighpass(img , D0):
	x , y = img.shape
	return Filter(img , GaussianFilter(x , y , D0 , highpass = True))

def GaussianLowpass(img , D0):
	x , y = img.shape
	return Filter(img , GaussianFilter(x , y , D0 , highpass = False))

def IdealHighpass(img , D0):
	x , y = img.shape
	return Filter(img , IdealFilter(x , y , D0 , highpass = True))

def IdealLowpass(img , D0):
	x , y = img.shape
	return Filter(img , IdealFilter(x , y , D0 , highpass = False))

def hybridImage(img1, img2, img1D0, img2D0 , Gaussian = True):
	if Gaussian == True:
		highPassed = GaussianHighpass(img1, img1D0)
		lowPassed = GaussianLowpass(img2, img2D0)
		return highPassed + lowPassed
	else:
		highPassed = IdealHighpass(img1, img1D0)
		lowPassed = IdealLowpass(img2, img2D0)
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
	filter_back = np.zeros(img.shape , dtype = np.float64)
	filter_back[:,:,0] = ShiftImage(np.fft.ifft2(filtered_f0).real)
	filter_back[:,:,1] = ShiftImage(np.fft.ifft2(filtered_f1).real)
	filter_back[:,:,2] = ShiftImage(np.fft.ifft2(filtered_f2).real)
	filter_back = normalize(filter_back)
	return filter_back

def GaussianHighpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , GaussianFilter(x , y , D0 , highpass = True))

def GaussianLowpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , GaussianFilter(x , y , D0 , highpass = False))

def IdealHighpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , IdealFilter(x , y , D0 , highpass = True))

def IdealLowpass_Color(img , D0):
	x , y , z = img.shape
	return Filter_Color(img , IdealFilter(x , y , D0 , highpass = False))

def hybridImage_Color(img1, img2, img1D0, img2D0 , Gaussian = True):
	if Gaussian == True:
		highPassed = GaussianHighpass_Color(img1, img1D0)
		lowPassed = GaussianLowpass_Color(img2, img2D0)
		return normalize(highPassed + lowPassed)
	else:
		highPassed = IdealHighpass_Color(img1, img1D0)
		lowPassed = IdealLowpass_Color(img2, img2D0)
		return normalize(highPassed + lowPassed)

def normalize(img):
	for i in range(3):
		img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i]))
		img[:,:,i] = img[:,:,i] / np.max(img[:,:,i])
	return img

img1 = cv2.imread("5_fish.bmp" )
img2 = cv2.imread("5_submarine.bmp" )

Highpass_image = IdealHighpass_Color(img1 , 20)
Lowpass_image = IdealLowpass_Color(img2 , 20)
hybrid = hybridImage_Color(img1, img2, 10 , 10 , Gaussian = False)

cv2.imshow('hybrid' , (hybrid*255).astype(np.uint8))
#cv2.imwrite('Ideal_output5.jpg', (hybrid*255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
