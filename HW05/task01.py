import numpy as np
import get_data_path

import function as f
if __name__ == '__main__':
	print("")
	# get image
	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_data_path.get_image_path(image_path)
	
	# get tiny image
	labels = ["normalize", "w/o normalize"]
	norms = [True, False]
	accuracy_line = []
	knns = []
	for label, norm in zip(labels, norms):
		train_tiny_feature = f.get_tiny_image(train_image_paths, norm)
		test_tiny_feature = f.get_tiny_image(test_image_paths, norm)
		accuracy = []
		knn = []
		for e in range(1, 20, 2):
			test_predicts = f.nearest_neighbor_classifier(train_tiny_feature, train_labels, test_tiny_feature, k = e)
			count = 0
			for i in range(len(test_labels)):
				if test_predicts[i] == test_labels[i]:
					count += 1

			print(label, e, "nn accuracy: ", count / len(test_labels) * 100 , '%')
			accuracy.append(count / len(test_labels) * 100)
			knn.append(e)

		accuracy_line.append(accuracy)
		knns.append(knn)

	f.plot_acc(accuracy_line, knns, labels, title = 'task01')