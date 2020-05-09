import numpy as np
import random


# Find homography which map @p2 to @p1.
#
# Inputs:
# @p1: point set 1.
# @p2: point set 2.
#
# Return: Homography for @p1 and @p2.
def GetHomography(p1, p2):
    assert isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray)
    assert p1.shape[0] == p2.shape[0]
    
    # Build the normalization matrix.
    def get_normalization_matrix(points):
        x_mean, y_mean = np.mean(points, axis=0)
        var_x, var_y = np.var(points, axis=0)
        s_x, s_y = np.sqrt(2/var_x), np.sqrt(2/var_y)
        return np.array([[s_x,   0, -s_x*x_mean],
                         [  0, s_y, -s_y*y_mean],
                         [  0,   0,           1]]), \
               np.array([[1/s_x,     0, x_mean],
                         [    0, 1/s_y, y_mean],
                         [    0,     0,      1]])

    # Normalize p1.
    N_p1, N_inv_p1 = get_normalization_matrix(p1)
    p1 = np.hstack((p1, np.ones((p1.shape[0], 1)))).T
    p1 = N_p1.dot(p1).T
    # Normalize p2.
    N_p2, _ = get_normalization_matrix(p2)
    p2 = np.hstack((p2, np.ones((p2.shape[0], 1)))).T
    p2 = N_p2.dot(p2).T

    # Build the "P" matrix in the homogeneous equation in 02-camera p.73
    #
    # @P: The "P" matrix.
    P = np.zeros((len(p1)<<1, 9))
    P[::2, :3] = p2
    P[1::2, 3:6] = p2
    P[::2, 6:] = -p2 * p1[:, 0, None]
    P[1::2, 6:] = -p2 * p1[:, 1, None]

    # The homography @H is the last column of the right singular matrix "V" of P.
    # Please remind that we get the tranpose of "V" through np.linalg.svd. Thus,
    # @H is the last **row** of "V".
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    tmp = vh[-1].reshape((3, 3))
    H = N_inv_p1.dot(tmp).dot(N_p2)
    return H

# Implement RANSAC.
#
# Inputs:
# @p1: point set 1.
# @p2: point set 2.
#
# Return: Best fit homography between @p1 and @p2.
def RANSAC(p1, p2):
    assert isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray)
    assert p1.shape[0] == p2.shape[0]

    # Sample @n_samples pairs in each iteration.
    n_samples = int(p1.shape[0] * 0.1)
    # Total @n_iters iterations.
    outlier_ratio = 0.05
    n_iters = int(np.log(1 - 0.99) / np.log(1 - (1-outlier_ratio)**n_samples))

    inlier_threshold = 10.0
    best_homography = None
    best_inlier_ratio = 0.0
    for _ in range(n_iters):
        # Get sample pairs.
        rand_idx = random.sample(range(0, p1.shape[0]), n_samples)
        tmp_p1, tmp_p2 = p1[rand_idx], p2[rand_idx]

        # Get homography which map tmp_p2 to tmp_p1.
        H = GetHomography(tmp_p1, tmp_p2)

        # Map p2 to p1's coordinate by the homography we got.
        tmp_p2 = np.hstack((p2, np.ones((p2.shape[0], 1)))).T
        map_p2 = H @ tmp_p2
        map_p2 /= map_p2[2, :]
        map_p2 = map_p2[:2, :].T
        # Use square error.
        error = np.sum((p1 - map_p2) ** 2, axis=1)
        # Calculate inlier ratio according to the threshold.
        inlier_num = len(error[error < inlier_threshold])
        inlier_ratio = inlier_num / p1.shape[0]
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_homography = H
    
    return best_homography
