# Standard packages.
import cv2
import os
import numpy as np
import time

# My packages.
import sift as kpd
import ransac


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
    cv2.imshow('Keypoints detection', out_img)
    cv2.waitKey(0)

def MapKp2ToKp1(kp2, homography, shiftx=0, shifty=0):
    tmp_kp2 = np.hstack((kp2, np.ones((kp2.shape[0], 1)))).T
    map_kp2 = homography @ tmp_kp2
    map_kp2 /= map_kp2[2, :]
    map_kp2[0] += shifty
    map_kp2[1] += shiftx
    return map_kp2[:2, :].T

def StitchingImg2ToImg1(img1, img2, homography, _shiftx, _shifty):
    invhomography = np.linalg.inv(homography)
    x = img2.shape[0]
    y = img2.shape[1]
    imgpoint = np.array([[0,0],               # image's corners
                         [y,0], 
                         [0,x], 
                         [y,x]])

    # Project image_2's four corners to image_1
    imgpoint = np.hstack((imgpoint, np.ones((imgpoint.shape[0], 1)))).T
    map_img2_cor = homography @ imgpoint  
    map_img2_cor /= map_img2_cor[2, :]
    map_img2_cor[0] += _shifty
    map_img2_cor[1] += _shiftx
    map_img2_cor = map_img2_cor[:2, :].T
    # Boundary for projected @img2.
    minx = int(np.min(map_img2_cor[:, 1]))+1
    maxx = int(np.max(map_img2_cor[:, 1]))
    miny = int(np.min(map_img2_cor[:, 0]))+1
    maxy = int(np.max(map_img2_cor[:, 0]))
    outputsize_x = np.max([img1.shape[0], maxx])
    outputsize_y = np.max([img1.shape[1], maxy])
    shiftx = minx * -1 if minx < 0 else 0
    shifty = miny * -1 if miny < 0 else 0
    # Initialize the output image.
    stitching_img = np.zeros([outputsize_x + shiftx, outputsize_y + shifty, 3])
    stitching_img[shiftx:img1.shape[0] + shiftx, shifty:img1.shape[1] + shifty, :] = img1
    
    # Use back projection to fill the color in the projected @img2.
    p1 = np.array([[j, i, 1] for i in range(minx, maxx) for j in range(miny, maxy)]).T
    ori_points = invhomography @ p1
    ori_points /= ori_points[2]
    ori_points = ori_points[:2]
    
    uni = np.zeros((8, ori_points.shape[1]))
    # Prepare four nearest pixels for interpolation.
    uni[0] = ori_points[1].astype(np.int64) # x0
    uni[1] = uni[0] + 1                     # x1
    uni[2] = ori_points[0].astype(np.int64) # y0
    uni[3] = uni[2] + 1                     # y1
    # Save @ori_points and @p1 in @uni to speed up filtering values.
    # Filtering condition: (x0 > 0) & (x1 < x-1) & (y0 > 0) & (y1 < y-1)
    uni[4:6] = ori_points
    uni[6] = np.add(p1[0], shifty + _shifty) 
    uni[7] = np.add(p1[1], shiftx + _shiftx) 
    # Filter values which satisfy the filtering condition.
    uni = uni[:, uni[0] > 0]
    uni = uni[:, uni[1] < x-1]
    uni = uni[:, uni[2] > 0]
    uni = uni[:, uni[3] < y-1]
    # Get values in @img2 according to x0, x1, y0, y1.
    uni_0 = tuple(uni[0].astype(np.int64))  # x0
    uni_1 = tuple(uni[1].astype(np.int64))  # x1
    uni_2 = tuple(uni[2].astype(np.int64))  # y0
    uni_3 = tuple(uni[3].astype(np.int64))  # y1
    val = np.array([
        img2[(uni_0, uni_2)],
        img2[(uni_0, uni_3)],
        img2[(uni_1, uni_2)],
        img2[(uni_1, uni_3)]
    ])

    # Interpolation.
    x1_x = uni[1] - uni[5]  # x1 - x
    y1_y = uni[3] - uni[4]  # y1 - y
    y_y0 = uni[4] - uni[2]  # y - y0
    x_x0 = uni[5] - uni[0]  # x - x0
    stitching_img[(tuple(uni[7].astype(np.int64)), tuple(uni[6].astype(np.int64)))] = \
        val[0] * (x1_x * y1_y)[:, None] + val[1] * (x1_x * y_y0)[:, None] + \
        val[2] * (x_x0 * y1_y)[:, None] + val[3] * (x_x0 * y_y0)[:, None]

    #stitching_img = stitching_img[:outputsize_x, :outputsize_y, :]
    return stitching_img, shiftx, shifty


if __name__ == '__main__':
    # Path to your images.
    data_dir = 'data'
    img_name = [['1.jpg', '2.jpg'], ['hill1.JPG', 'hill2.JPG'],
                ['S1.jpg', 'S2.jpg'],
                ['test1.jpg', 'test2.jpg', 'test3.jpg']]
    img_pth = [
        [os.path.join(data_dir, _name) for _name in _name_list]
        for _name_list in img_name
    ]
    
    index = 0
    for _img_pth in img_pth:
        homography = np.identity(3)
        shiftx, shifty = 0, 0
        stitching_img = cv2.imread(_img_pth[0])
        for img_pth_id in range(len(_img_pth) - 1):
            img1 = cv2.imread(_img_pth[img_pth_id]) # stitching_img
            img2 = cv2.imread(_img_pth[img_pth_id + 1])
            # -----> Part 01
            # Interest points detection & feature description by SIFT

            # Get keypoints and their descriptor.
            # 
            # @kpX: Keypoints.
            # @desX: Descriptor.
            kp1, des1 = kpd.SIFT(img1)
            kp2, des2 = kpd.SIFT(img2)
            # <----- Part 01

            # -----> Part 02
            # Feature matching using ratio distance.
            match_id = GetMatchFeaturesID(des1, des2)
            kp1, kp2 = kp1[match_id[:, 0]], kp2[match_id[:, 1]]
            DrawMatchKeypoints(img1, img2, kp1, kp2)

            # <----- Part 02

            # -----> Part 03
            # Get homography which maps p2 to p1's coordinate.
            homography_now = ransac.RANSAC(kp1, kp2)
            homography = homography @ homography_now
            # Map p2 to p1's coordinate.
            map_kp2 = MapKp2ToKp1(kp2, homography, shiftx, shifty)
            DrawMatchKeypoints(stitching_img, img2, map_kp2, kp2)
            # <----- Part 03
            
            # <----- Part 04
            start_t = time.time()
            stitching_img, shiftx, shifty = StitchingImg2ToImg1(stitching_img,
                                                                img2,
                                                                homography,
                                                                shiftx,
                                                                shifty)
            stitching_img = stitching_img.astype(np.uint8)
            print('Time:', time.time() - start_t)
            # -----> Part 04

        # Show the result.    
        cv2.imshow('map_img' + str(index), stitching_img)
        cv2.imwrite(str(index)+'.jpg', stitching_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        index = index+1
