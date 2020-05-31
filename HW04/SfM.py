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

    ############################################################################
    # Step 01: Find correspondences.
    ############################################################################
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
    #
    # @_xn: matched points. Shape: (N, 2)
    match_id = kpd.GetMatchFeaturesID(des1, des2)
    _x1, _x2 = kp1[match_id[:, 0]], kp2[match_id[:, 1]]
    kpd.DrawMatchKeypoints(img1, img2, _x1, _x2)

    # Transform @_x1 and @_x2 to normalized coordinate @x1 and @x2.
    #
    # Normalized coordinate: file coordinate -> [-1, 1] x [-1, 1]
    # Please refer to p.53 in the slides for details.
    #
    # @xn: normalized points. Shape: (N, 3)
    # => @x1 = @T1 @_x1. Point format: [y1, x1, 1]
    # => @x2 = @T2 @_x2. Point format: [y2, x2, 1]
    T1 = np.array([[2. / img1.shape[1],                  0, -1],
                   [                 0, 2. / img1.shape[0], -1],
                   [                 0,                  0,  1]])
    T2 = np.array([[2. / img2.shape[1],                  0, -1],
                   [                 0, 2. / img2.shape[0], -1],
                   [                 0,                  0,  1]])
    x1 = (T1 @ np.concatenate((_x1.T, np.ones((1, _x1.shape[0]))), axis=0)).T
    x2 = (T2 @ np.concatenate((_x2.T, np.ones((1, _x2.shape[0]))), axis=0)).T

    ############################################################################
    # Step 02: Find camera intrinsics.
    ############################################################################
    if img_name == 'Mesona':
        K1 = K2 = np.array([[1.4219, 0.0005, 0.5092],
                            [     0, 1.4219, 0.3802],
                            [     0,      0, 0.0010]])
        K1 /= K1[2, 2]
        K2 /= K2[2, 2]
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
    else:
        # TODO: Find camera intrinsics.
        print('Error!! No impelementation!!')
        import sys
        sys.exit(0)

    ############################################################################
    # Step 03: Using RANSAC to find the fundamental matrix
    ############################################################################
    F, inlier_idx = ransac_F.RANSAC(x1, x2)
    # de-normalize
    F = T1.T @ F @ T2
    _x1 = _x1[inlier_idx]
    _x2 = _x2[inlier_idx]
    SfMF.draw_epipolar_line(img1, img2, _x1, _x2, F)

    ############################################################################
    # Step 04: Get 4 possible solutions of essential matrix from fundamental
    # matrix.
    ############################################################################
    W = np.asarray([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
    Z = np.asarray([[ 0, 1, 0],
                    [-1, 0, 0],
                    [ 0, 0, 0]])
    # Get essential matrix @E.
    E = K1.T @ F @ K2
    # Make @E be rank 2.
    U, S, Vt = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    E = U @ np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]]) @ Vt
    U, S, Vt = np.linalg.svd(E)

    # Build the translation vector @t.
    Tx = U @ Z @ U.T
    t = np.asarray([Tx[2,1], Tx[0, 2], Tx[1, 0]]).reshape(-1, 1)

    # Build the rotation matrix @RX.
    R1 = U @ W @ Vt
    R1 = R1 * np.sign(np.linalg.det(R1))
    R2 = U @ W.T @ Vt
    R2 = R2 * np.sign(np.linalg.det(R2))    

    # Since there are 2 rotation matri @R1, @R2 and 2 translation vector @t, -@t
    # , we have four possible combination for @P2.
    #
    # @PX: Camera extrinsic [R|t]. We let the camera coordinate of the 1st image
    #   become world coordinate. Therefore, @P1 = [I|0].
    P1 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = [np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)),
          np.hstack((R2, -t))]

    # Let @X[i] be points in the world coordinate seen by both images.
    # => _x1 = K1 P1 X => (K1^-1) _x1 = P1 X
    # => _x2 = K2 P2 X => (K2^-1) _x2 = P2 X
    # Please refer to p.128 in the slides for more details.
    #
    # @xN_hat = (KN^-1) _xN. Shape: (3, N)
    x1_hat = K1_inv @ np.concatenate((_x1.T, np.ones((1, _x1.shape[0]))), axis=0)
    x1_hat /= x1_hat[2]
    x2_hat = K2_inv @ np.concatenate((_x2.T, np.ones((1, _x2.shape[0]))), axis=0) 
    x2_hat /= x2_hat[2]
    # We have @x1_hat = @P2 @x2_hat. Let @x2_hat be world coordinate @X. We get
    # @x2_hat = @P1 @X = @X. Then, we get @x1_hat = @P2 @x2_hat = @P2 @X.
    # We use the following relation to solve @X:
    # => @x1_hat = @P2 @X
    # => @x2_hat = @P1 @X
    # Please refer to p.128 in the slides for more details.
    X = [SfMF.triangulation(x1_hat, x2_hat, _P, P1) for _P in P2]

    # Choose @P2[i] which makes more points in front of the camera.
    # Please refere to p.130-131 in the slides for more details.
    #
    # @distance_X: The distance to the origin of the camera in world coordinate.
    distance_1 = [_X[:, 2] for _X in X]
    distance_2 = np.array([
        (X[i][:, :3] + P2[i][:, :3] @ P2[i][:, 3]) @ P2[i][2, :3].T
        for i in range(len(P2))
    ])

    # @scores: Number of points lie in front of the camera.
    scores = []
    for i in range(len(distance_1)):
        depth1 = distance_1[i]
        depth2 = distance_2[i]
        score = (depth1 > 0).astype(np.uint8) + (depth2 > 0).astype(np.uint8)
        scores.append(np.sum(score == 2))
    print(scores)

    # Choose the correct 3D points @X[i].
    correct_Rt_idx = np.argmax(scores)
    X = X[correct_Rt_idx]

    # Save data for matlab code.
    np.savetxt('meta/3d_points.txt', X[:, :3])
    np.savetxt('meta/2d_img_points.txt', _x1)
    np.savetxt('meta/camera_matrix.txt', np.hstack((K1, P1)))

    # Show 3D points.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    # @X[[i] has format [Y, X, Z], so we need to change the order.
    ax.scatter3D(X[:, 1], X[:, 0], X[:, 2])
    plt.show()
