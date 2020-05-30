# Standard packages.
import cv2
import numpy as np

# My packages.
import keypoint_detection as kpd
import ransac_F
import SfM_function as SfMF

if __name__ == '__main__':
    # Assumption: @x1 are points from first image. @x2 are points from the other
    # image.

    # Step 01: Find correspondences.
    #
    # Find @_x1 and @_x2.

    # Get keypoints and their descriptor by SIFT.
    # 
    # @kpX: Keypoints.
    # @desX: Descriptor.
    img_name = 'Mesona'
    img1 = cv2.imread(f'data/{img_name}1.JPG')
    img2 = cv2.imread(f'data/{img_name}2.JPG')
    kp1, des1 = kpd.SIFT(img1)
    kp2, des2 = kpd.SIFT(img2)

    # Feature matching using ratio distance.
    match_id = kpd.GetMatchFeaturesID(des1, des2)
    _x1, _x2 = kp1[match_id[:, 0]], kp2[match_id[:, 1]]
    kpd.DrawMatchKeypoints(img1, img2, _x1, _x2)
    # Transform @_x1 and @_x2 to normalized coordinate @x1 and @x2.
    #
    # Normalized coordinate: file coordinate -> [-1, 1] x [-1, 1]
    # Please refer to p.53 in the slides for details.
    #
    # => @x1 = @T1 @_x1, @x1 = [y1, x1, 1]
    # => @x2 = @T2 @_x2, @x2 = [y2, x2, 1]
    T1 = np.array([[2. / img1.shape[1],                  0, -1],
                   [                 0, 2. / img1.shape[0], -1],
                   [                 0,                  0,  1]])
    T2 = np.array([[2. / img2.shape[1],                  0, -1],
                   [                 0, 2. / img2.shape[0], -1],
                   [                 0,                  0,  1]])
    x1 = T1 @ np.concatenate((_x1.T, np.ones((1, _x1.shape[0]))), axis=0) 
    x2 = T2 @ np.concatenate((_x2.T, np.ones((1, _x2.shape[0]))), axis=0)
    # Step 02: Find camera intrinsics.
    #
    # Find @K1 and @K2.
    if img_name == 'Mesona':
        K1 = K2 = np.array([[1.4219, 0.0005, 0.5092],
                            [     0, 1.4219, 0.3802],
                            [     0,      0, 0.0010]])
    else:
        # TODO: Find camera intrinsics.
        print('Error!! No impelementation!!')
        import sys
        sys.exit(0)

    ############################################################################
    # Using RANSAC to find the fundamental matrix
    ############################################################################
    # write above parameters ans below for loop in ransac_F.RANSAC 
    F = ransac_F.RANSAC(x1.T, x2.T)
    # de-normalize
    F = T1.T @ F @ T2
    SfMF.draw_epipolar_line(img1, img2, _x1, _x2, F)
    ############################################################################
    # End RANSAC
    ############################################################################

    # Step 04: Get 4 possible solutions of essential matrix from fundamental
    # matrix.

    W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    E = K1.T @ F @ K2

    U, S, V = np.linalg.svd(E)

    m = (S[0] + S[1]) / 2
    E = U @ np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]]) @ V
    U, S, V = np.linalg.svd(E)

    #t = U[:, 2].reshape(-1, 1)
    Tx = U @ Z @ U.T
    t = np.asarray([Tx[2,1], Tx[0, 2], Tx[1, 0]]).reshape(-1, 1)

    R1 = U @ W @ V.T
    R1 = R1 * np.sign(np.linalg.det(R1))
    R2 = U @ W.T @ V.T
    R2 = R2 * np.sign(np.linalg.det(R2))    

    P1 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = [np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)),
          np.hstack((R2, -t))]

    X = [SfMF.triangulation(_x1.T, _x2.T, P1, _P) for _P in P2]

    x1 = [P1.dot(_X.T) for _X in X]
    x2 = [P2[i].dot(X[i].T) for i in range(len(P2))]

    scores = []
    for i in range(len(x1)):
        depth1 = x1[i][2,:]
        depth2 = x2[i][2,:]
        score = (depth1 > 0).astype(np.uint8) + (depth2 > 0).astype(np.uint8)
        scores.append(np.sum(score == 2))
    print(scores)
    P2 = P2[np.argmax(scores)]
