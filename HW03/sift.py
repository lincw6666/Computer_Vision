import cv2
import numpy as np


def SIFT(img_pth, debug=False):
    img = cv2.imread(img_pth)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray_img, None)

    # Show the keypoints in the image.
    if debug:
        img = cv2.drawKeypoints(gray_img, keypoints, img)
        cv2.imshow('Keypoints detection',img)
        cv2.waitKey(0)

    # Return a np.ndarray.
    return cv2.KeyPoint_convert(keypoints), descriptor


def Debug():
    SIFT('data/1.jpg', debug=True)


if __name__ == '__main__':
    Debug()
