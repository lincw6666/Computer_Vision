import cv2
import numpy as np


def SIFT(img, debug=False):
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


def SquareDistance(f1, f2):
    assert isinstance(f1, np.ndarray) and isinstance(f2, np.ndarray)
    assert f1.ndim == f2.ndim and f1.shape[-1] == f2.shape[-1]
    return np.sqrt(np.sum((f1 - f2) ** 2, axis=f1.ndim-1))


# Get the id of matched keypoints between @des1 and @des2 according to ratio
# distance.
#
# Inputs:
# @des1: Descriptor 1 for N points.
# @des2: Descriptor 2 for M points.
#
# Return: The id of matched keypoints.
#   [[keypoint1_id, keypoint2_id],
#                 ...
#                 ...
#    [keypoint1_id, keypoint2_id]]
def GetMatchFeaturesID(des1, des2):
    return np.asarray([
        [f1_id, f2_id[0]]
        for f1_id in range(des1.shape[0])
            for _dist in SquareDistance(des1[f1_id][None, :], des2)[None, :]
                for f2_id in np.argpartition(_dist, 2)[:2][None, :]
        if _dist[f2_id[0]] / _dist[f2_id[1]] < 0.6
    ])


def DrawMatchKeypoints(img1, img2, kp1, kp2):
    # Convert the type of matched keypoints to cv2.KeyPoint.
    p1 = [cv2.KeyPoint(x=_p[0], y=_p[1], _size=1) for _p in kp1]
    p2 = [cv2.KeyPoint(x=_p[0], y=_p[1], _size=1) for _p in kp2]

    # Build matched relationship in cv2.DMatch type.
    d_match = [
        cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0)
        for idx in range(len(p1))
    ]
    out_img = np.array([])
    out_img = cv2.drawMatches(img1, p1,
                              img2, p2,
                              d_match, out_img)
    cv2.namedWindow('Keypoints detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoints detection', out_img)
    cv2.waitKey(0)


def Debug():
    SIFT(cv2.imread('data/Mesona1.jpg'), debug=True)


if __name__ == '__main__':
    Debug()
