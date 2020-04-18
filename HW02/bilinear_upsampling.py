import numpy as np
import cv2


def bilinear_upsampling(img):
    ret_img = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]))

    # Origin image:
    # ---------
    # | a | b |
    # ---------
    # | c | d |
    # ---------
    # 
    # Bilinear upsampling image:
    # There are 9 cases:
    # ------------------
    # | 1 | 2 | 1 | 5 |
    # ------------------
    # | 3 | 4 | 3 | 7 |
    # ------------------
    # | 1 | 2 | 1 | 5 |
    # ------------------
    # | 6 | 8 | 6 | 9 |
    # ------------------

    # Case 01:
    # ------------------
    # | a |   | b |   |
    # ------------------
    # |   |   |   |   |
    # ------------------
    # | c |   | d |   |
    # ------------------
    # |   |   |   |   |
    # ------------------
    ret_img[::2, ::2] = img

    # Case 02 + Case 05:
    # ---------------------
    # |   |(a+b)/2|   | b |
    # ------------------
    # |   |       |   |   |
    # ------------------
    # |   |(c+d)/2|   | b |
    # ------------------
    # |   |       |   |   |
    # ---------------------
    ret_img[1::2, ::2] = img    # This action includes case 05.
    ret_img[1:-1:2, ::2] += img[1:, :]
    ret_img[1:-1:2, ::2] = ret_img[1:-1:2, ::2] / 2

    # Case 03 + Case 06:
    # -------------------------
    # |       |   |       |   |
    # -------------------------
    # |(a+c)/2|   |(b+d)/2|   |
    # -------------------------
    # |       |   |       |   |
    # -------------------------
    # |   c   |   |   d   |   |
    # -------------------------
    ret_img[::2, 1::2] = img    # This action includes case 06.
    ret_img[::2, 1:-1:2] += img[:, 1:]
    ret_img[::2, 1:-1:2] = ret_img[::2, 1:-1:2] / 2

    # Case 04 + Case 07 + Case 08 + Case 09:
    # -----------------------------
    # |   |           |   |       |
    # -----------------------------
    # |   |(a+b+c+d)/4|   |(c+d)/2|
    # -----------------------------
    # |   |           |   |       |
    # -----------------------------
    # |   |  (b+d)/2  |   |   d   |
    # -----------------------------
    ret_img[1::2, 1::2] = img   # case 04, 07, 08, 09.
    ret_img[1:-1:2, 1::2] += img[1:, :] # case 04, 07.
    ret_img[1::2, 1:-1:2] += img[:, 1:] # case 04, 08.
    ret_img[1:-1:2, 1:-1:2] += img[1:, 1:]  # case 04.
    ret_img[1:-1:2, 1:-1:2] = ret_img[1:-1:2, 1:-1:2] / 4   # case 04.
    ret_img[1:-1:2, -1] = ret_img[1:-1:2, -1] / 2   # case 07.
    ret_img[-1, 1:-1:2] = ret_img[-1, 1:-1:2] / 2   # case 08.
    return ret_img.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread('hw2_data/task1and2_hybrid_pyramid/1_bicycle.bmp')
    print(img.shape)
    upsample_img = bilinear_upsampling(img)
    print(upsample_img.shape)
    cv2.imshow('Test', upsample_img)
    cv2.waitKey(0)
