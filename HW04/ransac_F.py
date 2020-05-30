import numpy as np
import random


# Find homography which map @p2 to @p1.
#
# Inputs:
# @p1: point set 1.
# @p2: point set 2.
#
# Return: Homography for @p1 and @p2.

def GetFundamental(x1, x2):
    # (x1,x2 : 3*n arrays) using the 8 point algorithm
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    
    return F/F[2,2]

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
    inlier_threshold = 1e-3
    best_Fundamental = None
    best_inlier_ratio = 0.0
    best_inlier_idx = None
    for _ in range(n_iters):
        # Get sample pairs.
        rand_idx = random.sample(range(0, p1.shape[0]), n_samples)
        tmp_p1, tmp_p2 = p1[rand_idx], p2[rand_idx]

        # Get Fundamental
        F = GetFundamental(tmp_p1.T, tmp_p2.T)

        # Compute x^T F x for all correspondences 
        error = ( np.diag(np.dot(p1,np.dot(F, p2.T))) ) ** 2
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