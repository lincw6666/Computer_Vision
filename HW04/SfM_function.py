import numpy as np
import cv2
import random

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

		X = V[-1]
		points_3D.append(X/ X[3])

	return np.asarray(points_3D)

def comput_epipolar_line_slope(pts, img_number, F):
	#if img_number = 1 mean input point is from img1, Fundamental Matrix need to transpose
	if img_number == 1:
		f = F.T
	else :
		f = F
	a, b, c = f @ pts
	nu = a*a + b*b
	nu = 1/(nu**0.5) if nu else 1
	a*= nu
	b*= nu
	c*= nu 
	# return Linear equation ax + by + c = 0
	return np.array([a, b, c])

def draw_epipolar_line(img1, img2, _x1, _x2, F):
	x1 = np.concatenate((_x1.T, np.ones((1, _x1.shape[0]))), axis=0)
	x2 = np.concatenate((_x2.T, np.ones((1, _x2.shape[0]))), axis=0)
	'''
	N = 30 #random get N points
	R = random.sample(range(1, x1.shape[1]), N)
	x1 = x1[:, R]
	x2 = x2[:, R]
	'''
	h ,w, d = img1.shape
	for i in range(x1.shape[1]): 
		color = tuple(np.random.randint(0, 255, 3).tolist())
		l1 = comput_epipolar_line_slope(x2[:, i], 2, F)
		l2 = comput_epipolar_line_slope(x1[:, i], 1, F)

		#Calculate the point of the line at the boundary of the graph
		p1 = map(int, [0, -l1[2] / l1[1]])
		p2 = map(int, [w, -(l1[2] + l1[0] * w)/ l1[1]])
		p3 = map(int, [0, -l2[2] / l2[1]])
		p4 = map(int, [w, -(l2[2] + l2[0] * w)/ l2[1]])

		img1 = cv2.line(img1, tuple(p1), tuple(p2), color, 1)
		img2 = cv2.line(img2, tuple(p3), tuple(p4), color, 1)
		cv2.circle(img1, (int(x1[:, i][0]), int(x1[:, i][1])), 5, color, -1)
		cv2.circle(img2, (int(x2[:, i][0]), int(x2[:, i][1])), 5, color, -1)

	cv2.imshow('epopolar image1', img1)
	cv2.imshow('epopolar image2', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()