# Standard packages.
import os
import numpy as np

def get_image_path(image_path):

	train_image_paths = []
	test_image_paths = []

	train_labels = []
	test_labels = []

	categories = os.listdir(image_path + '/train')
	
	for category in categories:

		train_path = image_path + '/train/' + category + '/'
		train_image_names = os.listdir(train_path)
		for image in train_image_names:
			if not image.endswith('.jpg'):
				continue
			train_image_paths.append(train_path + image)
			train_labels.append(category)

		test_path = image_path + '/test/' + category + '/'
		test_image_names = os.listdir(test_path)
		for image in test_image_names:
			if not image.endswith('.jpg'):
				continue
			test_image_paths.append(test_path + image)
			test_labels.append(category)

	return train_image_paths, test_image_paths, train_labels, test_labels

if __name__ == '__main__':
	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_image_path(image_path)
	print(len(train_image_paths), len(test_image_paths), len(train_labels), len(test_labels))