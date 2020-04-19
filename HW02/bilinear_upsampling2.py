import numpy as np
import cv2


def bilinear_upsampling(img):
    ret_img = np.zeros((img.shape[0]*2 , img.shape[1]*2))

    np.set_printoptions(suppress=True)
    print(img)
    rdx, rdy = 6, 3 
    x, y = 2, 1
    # Origin image:
    # -------------
    # | a | b | c |
    # -------------
    # | d | e | f |
    # -------------
    # | g | h | i |
    # -------------
    # Bilinear upsampling image:
    # split into 4 areas, left up, right up, left down, right down
    # -------------------------
    # | a |   |   | b |   | c |
    # -------------------------
    # |   |   |   |   |   |   |
    # -------------------------
    # |   |   |   |   |   |   |
    # -------------------------
    # | d |   |   | e |   | f |
    # -------------------------
    # |   |   |   |   |   |   |
    # -------------------------
    # | g |   |   | h |   | i |
    # -------------------------

    # Case right down:
    # -------------
    # | e |   | f |   
    # -------------
    # |   |   |   | 
    # -------------
    # | h |   | i |   
    # -------------

    #corner
    ret_img[rdx::2, rdy::2] = img[x:, y:]
    
    #around
    ret_img[rdx::2, rdy+1::2] = img[x::, y:-1:]
    ret_img[rdx::2, rdy+1::2] += img[x::, y+1::]
    ret_img[rdx::2, rdy+1::2] = ret_img[rdx::2, rdy+1::2] / 2

    ret_img[rdx+1::2, rdy::2] = img[x:-1:, y::]
    ret_img[rdx+1::2, rdy::2] += img[x+1::, y::] 
    ret_img[rdx+1::2, rdy::2] = ret_img[rdx+1::2, rdy::2] / 2

    #center
    ret_img[rdx+1::2, rdy+1::2] = ret_img[rdx:-1:2, rdy+1::2] + ret_img[rdx+2::2, rdy+1::2] \
                                + ret_img[rdx+1::2, rdy:-1:2] + ret_img[rdx+1::2, rdy+2::2]
    ret_img[rdx+1::2, rdy+1::2] = ret_img[rdx+1::2, rdy+1::2] / 4

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
    #ret_img[1::2, ::2] = img    # This action includes case 05.
    #ret_img[1:-1:2, ::2] += img[1:, :]
    #ret_img[1:-1:2, ::2] = ret_img[1:-1:2, ::2] / 2

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
    #ret_img[::2, 1::2] = img    # This action includes case 06.
    #ret_img[::2, 1:-1:2] += img[:, 1:]
    #ret_img[::2, 1:-1:2] = ret_img[::2, 1:-1:2] / 2

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
    #ret_img[1::2, 1::2] = img   # case 04, 07, 08, 09.
    #ret_img[1:-1:2, 1::2] += img[1:, :] # case 04, 07.
    #ret_img[1::2, 1:-1:2] += img[:, 1:] # case 04, 08.
    #ret_img[1:-1:2, 1:-1:2] += img[1:, 1:]  # case 04.
    #ret_img[1:-1:2, 1:-1:2] = ret_img[1:-1:2, 1:-1:2] / 4   # case 04.
    #ret_img[1:-1:2, -1] = ret_img[1:-1:2, -1] / 2   # case 07.
    #ret_img[-1, 1:-1:2] = ret_img[-1, 1:-1:2] / 2   # case 08.
    print(ret_img)
    return ret_img.astype(np.uint8)


if __name__ == '__main__':
    img = np.array([[1,10,100, 10, 1],[0.1, 0.01, 0.05, 0.01, 0.1], [1, 2, 3, 2, 1],[0.1, 0.01, 0.05, 0.01, 0.1], [1, 10, 100, 10, 1]])
    print(img.shape)
    upsample_img = bilinear_upsampling(img)
    print(upsample_img.shape)

    img = np.array([[1,10,100, 10],[0.1, 0.01, 0.05, 0.01], [1, 2, 3, 2],[0.1, 0.01, 0.05, 0.01]])
    print(img.shape)
    upsample_img = bilinear_upsampling(img)
    print(upsample_img.shape)
    #cv2.imshow('Test', upsample_img)
    #cv2.waitKey(0)
