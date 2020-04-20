import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from convolution import convolution
from bilinear_upsampling2 import bilinear_upsampling

def downsampling(img):
	
	return img[0 : img.shape[0]-1 : 2, 0 : img.shape[1]-1 : 2]

def image_pyramid(imgs, filename = 'image pyramid', isgray = False):
	pyramid = np.zeros((imgs[0].shape[0] , imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[2] ), dtype= imgs[0].dtype)
	pyramid[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0][:,:]

	top = 0
	right = imgs[0].shape[1]
	for e in range(1, len(imgs)):
		pyramid[top:top + imgs[e].shape[0], right:right+imgs[e].shape[1]] = imgs[e][:,:]
		top += imgs[e].shape[0]

	color = "gray" if isgray else "RGB"
	cv2.imwrite('./task02_Output/'+ filename + '_pyramid_' + color +'.jpg', pyramid)
	'''
	cv2.imshow("pyramid", pyramid)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

imgpath = './task1and2_hybrid_pyramid'
Pyramids_layer = 5
files = os.listdir(imgpath)
isgray = True

#Gaussian Filter
GF_size = 5
x, y = np.mgrid[ -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1, -1 * (GF_size-1)/2 : (GF_size-1)/2 + 1]
gaussian_kernel = np.exp(-( x**2 + y**2) )

#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

for file in files:
	gaussian_img = []
	Laplacian_img = []
	recovery_img = []
	magnitude_spectrum_img = []
	if file == '.DS_Store':
		continue
	filename, extension = file.split('.')
	img = cv2.imread(imgpath + '/' + file)
	if isgray:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(img.shape[0], img.shape[1], -1)


	for i in range(Pyramids_layer):
		GF_img = convolution(img, gaussian_kernel)
		gaussian_img.append(GF_img)

		img = downsampling(GF_img)
	gaussian_img.append(img)
	Laplacian_img.append(img)


	T = True
	for i in range(Pyramids_layer - 1, -1, -1):
		Laplacian = gaussian_img[i] - bilinear_upsampling(img, gaussian_img[i].shape)
		Laplacian_img.append(Laplacian)
		#cv2.imwrite('./task02_Output/'+ filename + '_laplacian_layer{}'.format(i) + '.' + extension, Laplacian)
		img = gaussian_img[i]

	for e in Laplacian_img:
		if isgray:
			#Fourier magnitude
			dft = cv2.dft(np.float32(e), flags=cv2.DFT_COMPLEX_OUTPUT)
			dft_shift = np.fft.fftshift(dft)

			magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
			(tempx, tempy) = magnitude_spectrum.shape
			magnitude_spectrum_img.append(magnitude_spectrum.reshape(tempx, tempy, 1))
		

	for i in range(Pyramids_layer, 0, -1):
		l_index = len(Laplacian_img) - i
		
		recovery = Laplacian_img[l_index]+ bilinear_upsampling(gaussian_img[i], Laplacian_img[l_index].shape)
		recovery_img.append(recovery)


	Laplacian_img.reverse()
	recovery_img.reverse()
	magnitude_spectrum_img.reverse()

	#'''
	#plt.subplots(figsize = (20, 10))
	l = len(Laplacian_img)
	for i in range(1, l+1):

		plt.subplot(1, 2, 1),plt.imshow(Laplacian_img[i-1].reshape(Laplacian_img[i-1].shape[:2]), cmap = 'gray')
		plt.title(''), plt.xticks([]), plt.yticks([])
		plt.subplot(1, 2, 2),plt.imshow(magnitude_spectrum_img[i-1].reshape(magnitude_spectrum_img[i-1].shape[:2]))#, cmap = 'gray')
		plt.title(''), plt.xticks([]), plt.yticks([])
		plt.show()
	#'''

	image_pyramid(gaussian_img, filename + '_gaussian', isgray)
	image_pyramid(Laplacian_img, filename + '_laplacian', isgray)
	image_pyramid(recovery_img, filename + '_recovery', isgray)
	
	if isgray:
		image_pyramid(magnitude_spectrum_img, filename + '_spectrum', isgray)

