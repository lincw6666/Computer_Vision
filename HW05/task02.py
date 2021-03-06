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

	# get class_train_sift_features  15 * 100 * n * 128
	if os.path.isfile('class_train_sift_features.npy'):
		class_train_sift_features = np.load('class_train_sift_features.npy', allow_pickle=True)
	else :
		bag_of_features = []
		class_train_sift_features = []
		i = 0
		for path in train_image_paths:
			img = cv2.imread(path)
			keypoints, descriptor = sift.SIFT(img)
			bag_of_features.append(descriptor)
			i += 1
			if i % 100 == 0:
				# bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
				class_train_sift_features.append(bag_of_features)
				bag_of_features = []
		class_train_sift_features = np.array(class_train_sift_features)
		np.save('class_train_sift_features.npy', class_train_sift_features)

	# get class_test_features  15 * 10 * n * 128
	if os.path.isfile('class_test_features.npy'):
		class_test_features = np.load('class_test_features.npy', allow_pickle=True)
	else :
		bag_of_features = []
		class_test_features = []
		i = 0
		for path in test_image_paths:
			img = cv2.imread(path)
			keypoints, descriptor = sift.SIFT(img)
			bag_of_features.append(descriptor)
			i += 1
			if i % 10 == 0:
				# bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
				class_test_features.append(bag_of_features)
				bag_of_features = []
		class_test_features = np.array(class_test_features)
		np.save('class_test_features.npy', class_test_features)

	# get sift feature of test image
	if os.path.isfile('test_sift_features.npy'):
		test_features = np.load('test_sift_features.npy')
	else :
		test_features = []
		for path in test_image_paths:
			img = cv2.imread(path)
			keypoints, descriptor = sift.SIFT(img)
			test_features.append(descriptor)

		test_features = np.concatenate(test_features, axis=0).astype('float32')
		np.save('test_sift_features.npy', test_features)

	if os.path.isfile('vocabulary_128.npy'):
		center = np.load('vocabulary_128.npy')
	else :
		center = f.k_means(bag_of_features, k = 256)
		np.save('vocabulary.npy', center)


	# get train_histogram = 15 * 100 * 128
	if os.path.isfile('train_histogram_128.npy'):
		train_histogram = np.load('train_histogram_128.npy')
	else :
		train_histogram = f.histogram(class_train_sift_features, center)
		np.save('train_histogram_128.npy', train_histogram)

	# get test_histogram = 15 * 10 * 128
	if os.path.isfile('test_histogram_128.npy'):
		test_histogram = np.load('test_histogram_128.npy')
	else :
		test_histogram = f.histogram(class_test_features, center)
		np.save('test_histogram_128.npy', test_histogram)

	train_histogram = train_histogram.reshape((-1,128))
	test_histogram = test_histogram.reshape((-1,128))
	for k_num in range(1,15):
		test_predicts = f.nearest_neighbor_classifier(train_histogram, train_labels, test_histogram, k = k_num)
		count = 0
		for i in range(150):
			if test_labels[i] == test_predicts[i]:
				count += 1
		print("k = ", k_num, " ", count/150)
