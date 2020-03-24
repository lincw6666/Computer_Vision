import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
"""
Write your code here
"""
# Find Homography.
#
# @H: A list contains the homography of each image to the chessboard.
H = []

# It takes a lot of time finding corner in the images. I saved the results to files
# you can simply skip finding corner and use the data in the files directly.
#np.save('imgpoints.npy', np.array(imgpoints))
imgpoints = np.load('imgpoints.npy')
#np.save('objpoints.npy', np.array(objpoints))
objpoints = np.load('objpoints.npy')


for p_imgs, p_objs in zip(imgpoints, objpoints):
    assert p_imgs.shape[0] == p_objs.shape[0], "Number of corner in the image"\
        "should match those on the real world chessboard."
    
    # Build the "P" matrix in the homogeneous equation in 02-camera p.73
    #
    # @P: The "P" matrix.
    P = np.zeros((len(p_imgs)<<1, 9))
    p_objs[:, 2] = 1
    P[::2, :3] = p_objs
    P[1::2, 3:6] = p_objs
    P[::2, 6:] = -p_objs * p_imgs[:, 0, 0, None]
    P[1::2, 6:] = -p_objs * p_imgs[:, 0, 1, None]

    # The homography @H[i] is the last column of the right singular matrix "V" of P.
    # Please remind that we get the tranpose of "V" through np.linalg.svd. Thus,
    # @H[i] is the last **row** of "V".
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    H.append(vh[-1].reshape((3, 3)))

    # Verify whether we get the correct homography.
    """
    print('\n**********************************************')
    for p_img, p_obj in zip(p_imgs[:, 0, :], p_objs):
       get_point = H[-1].dot(p_obj)
       get_point = get_point[:2] / get_point[2]
       print(p_img, get_point)
    """
   
    
# Find intrinsic matrix K
    
def buildv(H, i, j):
    v = [H[0][i-1] * H[0][j-1], 
         H[0][i-1] * H[1][j-1] + H[1][i-1] * H[0][j-1], 
         H[1][i-1] * H[1][j-1],
         H[2][i-1] * H[0][j-1] + H[0][i-1] * H[2][j-1], 
         H[2][i-1] * H[1][j-1] + H[1][i-1] * H[2][j-1], 
         H[2][i-1] * H[2][j-1]]
    return np.array(v)

V = []
for i in range(len(H)):
    V.append(buildv(H[i], 1, 2))
    V.append(buildv(H[i], 1, 1) - buildv(H[i], 2, 2))
V = np.array(V)

_, _, vh = np.linalg.svd(V)
b = vh[-1]
B = np.array([[b[0], b[1], b[3]],
              [b[1], b[2], b[4]],
              [b[3], b[4], b[5]]])
K = np.linalg.cholesky(B)
K = np.linalg.inv(K)
K = K.T
    
    
# get extrinsic matrix [R|t] for each images by K and H 

extrinsics = np.array([], dtype=np.float32).reshape(3,0)
K_inverse = np.linalg.inv(K)

for h in H:
    h1,h2,h3 = [h[:,e] for e in range(3)]
    lambda_ = 1/(np.linalg.norm(K_inverse.dot(h1)))

    r1 = lambda_ * K_inverse.dot(h1)
    r2 = lambda_ * K_inverse.dot(h2)
    r3 = np.cross(r1,r2)
    t = lambda_ * K_inverse.dot(h3)

    R = np.hstack((r1.reshape(3,1),r2.reshape(3,1),r3.reshape(3,1)))

    extrinsics = np.hstack((extrinsics,R))
    extrinsics = np.hstack((extrinsics,t.reshape(3,1)))
    print(extrinsics.shape)

print(extrinsics.shape)
extrinsics = extrinsics.reshape(-1,3,4)



import sys
#sys.exit(0)
"""
"""
# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K#mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
