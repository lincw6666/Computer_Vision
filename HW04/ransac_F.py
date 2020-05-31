import numpy as np
import random


# Find homography which map @x2 to @x1.
#
# Inputs:
# @x1: point set 1. Point format: [y, x, 1]
# @x2: point set 2. Point format: [y, x, 1]
#
# Return: Fundamental matrix from @x2 to @x1.
def GetFundamental(x1, x2):
    # (x1,x2 : n*3 arrays) using the 8 point algorithm
    n = x1.shape[0]
    if x2.shape[0] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0]*x2[i,2],
                x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1]*x2[i,2],
                x1[i,2]*x2[i,0], x1[i,2]*x2[i,1], x1[i,2]*x2[i,2] ]
            
    # compute linear least square solution
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    
    return F / F[2,2]

# Implement RANSAC.
#
# Inputs:
# @p1: point set 1.
# @p2: point set 2.
#
# Return: Best fit fundamental matrix from @p2 to @p1.
def RANSAC(p1, p2):
    assert isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray)
    assert p1.shape[0] == p2.shape[0]

    # Sample @n_samples pairs in each iteration.
    n_samples = int(p1.shape[0] * 0.1)
    # Total @n_iters iterations.
    outlier_ratio = 0.05
    n_iters = int(np.log(1 - 0.99) / np.log(1 - (1-outlier_ratio)**n_samples))
    inlier_threshold = 1e-3
    best_Fundamental = None
    best_inlier_ratio = 0.0
    best_inlier_idx = None
    for _ in range(n_iters):
        # Get sample pairs.
        rand_idx = random.sample(range(0, p1.shape[0]), n_samples)
        tmp_p1, tmp_p2 = p1[rand_idx], p2[rand_idx]

        # Get Fundamental
        F = GetFundamental(tmp_p1, tmp_p2)

        # Compute |x^T F x|^2 for all correspondences 
        error = (np.diag(p1 @ F @ p2.T)) ** 2
        # Use square error.
        # error = np.sqrt(np.sum((p1 - map_p2) ** 2, axis=1))
        # Calculate inlier ratio according to the threshold.
        inlier_idx = np.where(error < inlier_threshold)
        inlier_num = len(inlier_idx[0])
        inlier_ratio = inlier_num / p1.shape[0]
        if inlier_ratio >= best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_Fundamental = F
            best_inlier_idx = inlier_idx
    return best_Fundamental, best_inlier_idx