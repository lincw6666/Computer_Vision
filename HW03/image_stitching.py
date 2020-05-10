# Standard packages.
import cv2
import os
import numpy as np

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

def DrawMatchKeypoints(img_pth1, img_pth2, kp1, kp2):
    # Convert the type of matched keypoints to cv2.KeyPoint.
    p1 = [cv2.KeyPoint(x=_p[0], y=_p[1], _size=1) for _p in kp1]
    p2 = [cv2.KeyPoint(x=_p[0], y=_p[1], _size=1) for _p in kp2]

    # Build matched relationship in cv2.DMatch type.
    d_match = [
        cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0)
        for idx in range(len(p1))
    ]
    out_img = np.array([])
    out_img = cv2.drawMatches(cv2.imread(img_pth1), p1,
                                cv2.imread(img_pth2), p2,
                                d_match, out_img)
    cv2.imshow('Keypoints detection', out_img)
    cv2.waitKey(0)

def MapKp2ToKp1(kp2, homography):
    tmp_kp2 = np.hstack((kp2, np.ones((kp2.shape[0], 1)))).T
    map_kp2 = homography @ tmp_kp2
    map_kp2 /= map_kp2[2, :]
    return map_kp2[:2, :].T

def StitchingImg2ToImg1(img_pth1, img_pth2, homography):
    img1 = cv2.imread(img_pth1)
    img2 = cv2.imread(img_pth2)
    invhomography = np.linalg.inv(homography)
    x = img2.shape[0]
    y = img2.shape[1]
    imgpoint = np.array([[0,0],               # image's corners
                         [y,0], 
                         [0,x], 
                        [y,x]])
    imgpoint = np.hstack((imgpoint, np.ones((imgpoint.shape[0], 1)))).T
    # project image_2's four corners to image_1
    map_img2_cor = homography @ imgpoint       
    map_img2_cor /= map_img2_cor[2, :]
    map_img2_cor = map_img2_cor[:2, :].T
    minx = int(np.min(map_img2_cor[:,1]))+1
    maxx = int(np.max(map_img2_cor[:,1]))
    miny = int(np.min(map_img2_cor[:,0]))+1
    maxy = int(np.max(map_img2_cor[:,0]))
    outputsize_x = np.max([x, maxx])
    outputsize_y = np.max([y, maxy])
    stitching_img = np.zeros([outputsize_x*2,outputsize_y*2,3])
    stitching_img[:x,:y,:] = img1
    for i in range(minx,maxx):
        for j in range(miny,maxy):
            # inverse project the point to image_2
            ori_point = np.dot(invhomography, np.array([j,i,1]).T)
            ori_point /= ori_point[2]
            ori_point = ori_point[:2].T
            # determine the near four points to use linear interpolation
            x0 = int(ori_point[1])
            x1 = int(ori_point[1])+1
            y0 = int(ori_point[0])
            y1 = int(ori_point[0])+1
            if (x0 > 0) & (x1 < x-1) & (y0 > 0) & (y1 < y-1):
                for z in range(3):
                    near_points = [(x0 , y0 , img2[x0,y0,z]), (x0 , y1 , img2[x0,y1,z]), (x1 , y0 , img2[x1,y0,z]), (x1 , y1 , img2[x1,y1,z])]
                    stitching_img[i,j,z] = interpolation(ori_point[1], ori_point[0], near_points)
    stitching_img = stitching_img[:outputsize_x, :outputsize_y, :]
    return stitching_img


if __name__ == '__main__':
    # Path to your images.
    data_dir = 'data'
    img_name = [['1.jpg', '2.jpg'], ['hill1.JPG', 'hill2.JPG'],
                ['S1.jpg', 'S2.jpg']]
    img_pth = [
        [os.path.join(data_dir, _name[0]), os.path.join(data_dir, _name[1])]
        for _name in img_name
    ]
    
    index = 1
    for img_pth1, img_pth2 in img_pth:
        # -----> Part 01
        # Interest points detection & feature description by SIFT

        # Get keypoints and their descriptor.
        # 
        # @kpX: Keypoints.
        # @desX: Descriptor.
        kp1, des1 = kpd.SIFT(img_pth1)
        kp2, des2 = kpd.SIFT(img_pth2)
        # <----- Part 01

        # -----> Part 02
        # Feature matching using ratio distance.
        match_id = GetMatchFeaturesID(des1, des2)
        kp1, kp2 = kp1[match_id[:, 0]], kp2[match_id[:, 1]]
        DrawMatchKeypoints(img_pth1, img_pth2, kp1, kp2)

        # <----- Part 02

        # -----> Part 03
        # Get homography which maps p2 to p1's coordinate.
        homography = ransac.RANSAC(kp1, kp2)
        # Map p2 to p1's coordinate.
        map_kp2 = MapKp2ToKp1(kp2, homography)
        DrawMatchKeypoints(img_pth1, img_pth2, map_kp2, kp2)
        # <----- Part 03
        
        # <----- Part 04
        stitching_img = StitchingImg2ToImg1(img_pth1, img_pth2, homography)
        cv2.imshow('map_img' + str(index), stitching_img.astype(np.uint8))
        cv2.imwrite(str(index)+'.jpg', stitching_img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        index = index+1
        # -----> Part 04
