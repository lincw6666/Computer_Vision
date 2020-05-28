# Standard packages.
import cv2
import numpy as np

# My packages.
import keypoint_detection as kpd


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

    # RANSAC parameters.
    #
    # Sample @n_samples pairs in each iteration.
    n_samples = 8
    # Total @n_iters iterations.
    outlier_ratio = 0.05
    n_iters = int(np.log(1 - 0.99) / np.log(1 - (1-outlier_ratio)**n_samples))

    inlier_threshold = 10.0 # It's not the correct value. Need to be modify.
    best_F = None
    best_inlier_ratio = 0.0

    for _ in range(n_iters):
        # Sample @n_samples points.

        #
        # TODO
        #

        # Step 03: Find fundamental matrix using 8 points algorithm. Please
        # refer to p.46 in the slides for details.
        #
        # @F: @F = (K1^-T) @E (K2^-1) s.t. (x1^T) @F x2 = 0.
        #   1. @F x2 => epipolar line associated with x1
        #   2. (@F^T) x1 => epipolar line associated with x2
        #   3. @F e2 = 0
        #   4. (@F^T) e1 = 0
        #   5. det(@F) = 0, @F has rank 2
        
        # Build equation Af = 0.
        # ----->
        # Please finish the above TODO first.
        # <-----
        # A = np.hstack(([sample_x1.T] * 3))
        # A[:, :3] = A[:, :3] * sample_x2[0].T.unsqueeze(-1)
        # A[:, 3:6] = A[:, 3:6] * sample_x2[1].T.unsqueeze(-1)

        # 1. Find @f from the last right singular vector of @A.
        # 2. Reshape @f to 3x3 to get @F.

        # 
        # TODO
        #

        # Resolve det(@F) = 0 constraint using SVD.
        # F = USV^T, where S[3, 3] = 0.

        # 
        # TODO
        #

        # Draw epipolar line and correspondences on image.

        #
        # TODO
        #

        # Using RANSAC to deal with outliers (sample 8 points).
        # => |x2 @F x1| < threshold

        #
        # TODO
        #

    ############################################################################
    # End RANSAC
    ############################################################################

    # Step 04: Get 4 possible solutions of essential matrix from fundamental
    # matrix.

    #
    # TODO
    #
