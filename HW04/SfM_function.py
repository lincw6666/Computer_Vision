import numpy as np


def triangulation(x1, x2, P1, P2):

	# X = [u v 1]T
	points_3D = []
	points_num = x1.shape[1]
	for i in range(points_num):
		a1 = x1[0, i] * P1[2, :] - P1[0, :]
		a2 = x1[1, i] * P1[2, :] - P1[1, :]
		a3 = x2[0, i] * P2[2, :] - P2[0, :]
		a4 = x2[1, i] * P2[2, :] - P2[1, :]

		A = [a1, a2, a3, a4]

		U, S, V = np.linalg.svd(A)

		X = V[:, -1]
		points_3D.append(X/ X[3])

	return np.asarray(points_3D)


def draw_epipolar_line(img1, img2, _x1, _x2, F):
	