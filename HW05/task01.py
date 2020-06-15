import numpy as np
import get_data_path

import function as f
if __name__ == '__main__':

	# get image
	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_data_path.get_image_path(image_path)
	
	# get tiny image
	train_tiny_feature = f.get_tiny_image(train_image_paths)
	test_tiny_feature = f.get_tiny_image(test_image_paths)

	for e in range(1, 15, 2):
		test_predicts = f.nearest_neighbor_classifier(train_tiny_feature, train_labels, test_tiny_feature, k = e)
		count = 0
		for i in range(len(test_labels)):
			if test_predicts[i] == test_labels[i]:
				count += 1

		print(e, "nn accuracy: ", count / len(test_labels) * 100 , '%')