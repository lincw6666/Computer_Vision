from PIL import Image
import numpy as np

import get_data_path

def get_tiny_image(image_paths):

	height, width = 16, 16
	tiny_images = np.zeros((len(image_paths), height * width))

	for e, path in enumerate(image_paths):

		image = Image.open(path)
		image_flatten = np.array(image.resize((width, height), Image.ANTIALIAS)).flatten()
		image_nm = (image_flatten - np.mean(image_flatten))/np.std(image_flatten)
		tiny_images[e, :] = image_nm
	
	return tiny_images


def nearest_neighbor_classifier(train_image_features, train_labels, test_image_features, k = 3):

	
	dist = np.zeros((len(test_image_features), len(train_image_features)))
	for i, test_f in enumerate(test_image_features):
		for j, train_f in enumerate(train_image_features):
			dist[i, j] = np.sum(np.square(test_f - train_f))

	#return [train_labels[i] for i in np.argmin(dist, axis = 1)] # k = 1
	test_predicts = []
	for i in range(dist.shape[0]):
		d = np.argsort(dist[i, :])

		vote = {}
		for j in range(k):
			if train_labels[d[j]] in vote.keys():
				vote[train_labels[d[j]]] += 1
			else :
				vote[train_labels[d[j]]] = 1

		neighbor = 0
		for j in vote.keys():
			if neighbor < vote[j]:
				neighbor = vote[j]
				predict = j
		test_predicts.append(predict)

	return test_predicts

if __name__ == '__main__':

	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_data_path.get_image_path(image_path)
	train_tiny_feature = get_tiny_image(train_image_paths)
	test_tiny_feature = get_tiny_image(test_image_paths)
	nearest_neighbor_classifier(train_tiny_feature, train_labels, test_tiny_feature)