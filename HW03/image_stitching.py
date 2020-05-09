# Standard packages.
import cv2
import os

# My packages.
import sift as kpd


if __name__ == '__main__':
    # Path to your images.
    data_dir = 'data'
    img_name = [['1.jpg', '2.jpg'], ['hill1.JPG', 'hill2.JPG'],
                ['S1.jpg', 'S2.jpg']]
    img_pth = [
        [os.path.join(data_dir, _name[0]), os.path.join(data_dir, _name[1])]
        for _name in img_name
    ]

    for img_pth1, img_pth2 in img_pth:
        # -----> Part01
        # Interest points detection & feature description by SIFT

        # Get keypoints and their descriptor.
        # 
        # @kpX: Keypoints.
        # @desX: Descriptor.
        kp1, des1 = kpd.SIFT(img_pth1)
        kp2, des2 = kpd.SIFT(img_pth2)
        # <----- Part01
