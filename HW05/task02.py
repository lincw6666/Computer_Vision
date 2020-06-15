import numpy as np
import cv2
import os

import get_data_path
import function as f
import sift
if __name__ == '__main__':

	# get image
	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_data_path.get_image_path(image_path)
	
	# get sift feature
	if os.path.isfile('train_sift_features.npy'):
		bag_of_features = np.load('train_sift_features.npy')
	else :
		bag_of_features = []
		for path in train_image_paths:
			img = cv2.imread(path)
			keypoints, descriptor = sift.SIFT(img)
			bag_of_features.append(descriptor)

		bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
		np.save('train_sift_features.npy', bag_of_features)

	if os.path.isfile('vocabulary.npy'):
		center = np.load('vocabulary.npy')
	else :
		center = f.k_means(bag_of_features, k = 5)
		np.save('vocabulary.npy', center)

	

	