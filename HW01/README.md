Homework 01 - Camera Calibration
===

# What We've done
- [x] Calculate the homography of each image taken from the camera to the real world chessboard.
- [x] Get the intrinsic matrix.
- [x] Get the extrinsic matrices.

# Code Description
## Get Homography
- Store the corners in each image to a file "imgpoints.npy"; store the corners of the real chessboard to a file "objpoints.npy". So you don't need to spend a lot of time finding corners. Just load them by "np.load". If you still want to run the code to find the corners, you can comment out "continue" in the loop.
- Solve the optimization problem *Ph = 0*, which is derived from *p_img = H p_world*.
  - *h* = the smallest right singular value of *P*
  - Reshape *h* in to *H*

## Get Intrinsic Matrix
- Solve *Vb = 0*, which is derived from *h1^T K^-T K^-1 h2 = 0* and *h1^T K^-T K^-1 h1 = h2^T K^-T K^-1 h2*, where *h1, h2* respect to the first and the second column of the homography *H*.
- Use *b* to build *B*, where *B* is a positive definite matrix equals to *K^-T K^-1*.
- We might get a negative definite *B*. Multiply *b* by -1 fix the problem since a singular vector is still a singular vector after multiplying any scaler.

## Get Extrinsic Matrix
- The rotation matrix *R = [r1 r2 r3]*
  - *r1 = K^-1 h1 / |r1|*
  - *r2 = K^-1 h2 / |r1|*
  - *r3 = r1 X r2*
  - Since *r1, r2, r3* have norm 1 and *H* might multiply by a scaler, we need to normalize *r1, r2* then we get a normalized *r3*.
- *t = K^-1 h3 / [r1]*
- The extrinsic matrix *E = [R|t]*

# Usage
Clone the repo and run `python camera_calibration.py (optional)dir_path_to_your_images`.
