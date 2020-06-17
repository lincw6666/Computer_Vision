from PIL import Image
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import cv2

import get_data_path

def get_tiny_image(image_paths, norm = False):

	height, width = 16, 16
	tiny_images = np.zeros((len(image_paths), height * width))

	for e, path in enumerate(image_paths):

		image = Image.open(path)
		image_flatten = np.array(image.resize((width, height), Image.ANTIALIAS)).flatten()
		#image = cv2.imread(path, 0)
		#image_flatten = cv2.resize(image, (width, height)).flatten()
		if norm:
			image_nm = (image_flatten - np.mean(image_flatten))/np.std(image_flatten)
			tiny_images[e, :] = image_nm
		else :
			tiny_images[e, :] = image_flatten
	
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

def getdist(p1, p2):
	return math.sqrt(np.sum(np.square(p1 - p2)))

def k_means(data, k):

	center_index = random.sample(list(range(len(data))), k)
	new_center = np.asarray([data[i] for i in center_index])
	center = np.zeros((new_center.shape))
	#cluster_index = np.zeros((len(data)))
	#cluster_new_index = np.ones((len(data)))
	#while cluster_new_index.all() != cluster_index.all():
	while True:
		#cluster_index  = copy.deepcopy(cluster_new_index)
		center = copy.deepcopy(new_center)
		C = [[] for i in range(k)]

		for i, item in enumerate(data):
			class_index = -1
			mindist = 2e9

			for j, center_point in enumerate(center):
				dist = getdist(item, center_point)
				if dist < mindist:
					class_index = j
					mindist = dist
			C[class_index].append(item)
			#cluster_new_index [i] = class_index
		
		for i, cluster in enumerate(C):
			new_center[i] = np.mean(np.asarray(cluster), axis = 0)

		issame = True
		for old, new in zip(center, new_center):
			for i, j in zip(old, new):
				if i != j:
					issame = False
		if issame:
			break
	return center

def plot_acc(accuracys, knns, labels, title = 'accuracy'):

	plt.title(title)
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	for acc, knn, label in zip(accuracys, knns, labels):
		plt.plot(knn, acc, label = label)

	plt.show()

def histogram(feature, center):
    # class_feature = 15 * 100 * n * 128  or 15 * 10 * n * 128
    # center = 128(k numbers) * 128(feature)
    #  histogram = 15 * 100 * 128
    k = center.shape[0]
    image_num = feature[0].shape[0]
    hist = np.zeros((15*image_num,k)).reshape((15,image_num,k))
    for i in range(15):
        for j in range(image_num):
            for feature_num in range(feature[i][j].shape[0]):
                hist_index = np.argmin(np.sqrt(np.sum(np.square(feature[i][j][feature_num] - center), 1)))
                hist[i][j][hist_index] += 1
    return hist	

if __name__ == '__main__':

	image_path = './hw5_data'
	train_image_paths, test_image_paths, train_labels, test_labels = get_data_path.get_image_path(image_path)
	train_tiny_feature = get_tiny_image(train_image_paths)
	#test_tiny_feature = get_tiny_image(test_image_paths)
	#nearest_neighbor_classifier(train_tiny_feature, train_labels, test_tiny_feature)

	k_means(train_tiny_feature, 5)
