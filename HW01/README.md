Homework 01 - Camera Calibration
===

# What We've done
- [x] Calculate the homography of each image taken from the camera to the real world chessboard.
- [ ] Get the intrinsic matrix.
- [ ] Get the extrinsic matrices.

# Code Description
## Get Homography
- Store the corners in each image to a file "imgpoints.npy"; store the corners of the real chessboard to a file "objpoints.npy". So you don't need to spend a lot of time finding corners. Just load them by "np.load". If you still want to run the code to find the corners, you can comment out "continue" in the loop.
- Comment out the calibration done by "cv2".
- **Remove "sys.exit(0)" before you do further works.**

## Get Intrinsic Matrix
Comming soon...

## Get Extrinsic Matrix
Comming soon...

# Usage
Clone the repo and run `python camera_calibration.py`.
