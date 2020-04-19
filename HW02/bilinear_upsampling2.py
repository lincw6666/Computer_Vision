import numpy as np
import cv2


def bilinear_upsampling(img, shape):

    ret_img = np.zeros((shape))

    if shape[0]%2 == 0:
        rdx, x = 3, 1
    else :
        rdx, x = 6, 2

    if shape[1]%2 == 0:
        rdy, y = 3, 1
    else :
        rdy, y = 6, 2

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
    # | e | 2 | f |   
    # -------------
    # | 2 | 3 | 2 | 
    # -------------
    # | h | 2 | i |   
    # -------------

    #corner
    ret_img[rdx::2, rdy::2] = img[x:, y:]
    
    #case 2
    ret_img[rdx::2, rdy+1::2] = img[x::, y:-1:]
    ret_img[rdx::2, rdy+1::2] += img[x::, y+1::]
    ret_img[rdx::2, rdy+1::2] = ret_img[rdx::2, rdy+1::2] / 2

    ret_img[rdx+1::2, rdy::2] = img[x:-1:, y::]
    ret_img[rdx+1::2, rdy::2] += img[x+1::, y::] 
    ret_img[rdx+1::2, rdy::2] = ret_img[rdx+1::2, rdy::2] / 2

    #case 3
    ret_img[rdx+1::2, rdy+1::2] = ret_img[rdx:-1:2, rdy+1::2] + ret_img[rdx+2::2, rdy+1::2] \
                                + ret_img[rdx+1::2, rdy:-1:2] + ret_img[rdx+1::2, rdy+2::2]
    ret_img[rdx+1::2, rdy+1::2] = ret_img[rdx+1::2, rdy+1::2] / 4

    # Case left up:
    # -----------------
    # | a | 1 | 2 | b |   
    # -----------------
    # | 1 | 3 |   | 1 | 
    # -----------------
    # | 2 |   |   | 2 |   
    # -----------------
    # | d | 1 | 2 | e |   
    # -----------------

    ret_img[0:rdx+1:3, 0:rdy+1:3] = img[:x+1, :y+1]

    # case 1
    ret_img[0:rdx+1:3, 1:rdy:3] = img[:x+1, :y] 
    ret_img[0:rdx+1:3, 1:rdy:3] += img[:x+1, :y] 
    ret_img[0:rdx+1:3, 1:rdy:3] += img[:x+1, 1:y+1] 
    ret_img[0:rdx+1:3, 1:rdy:3] = ret_img[0:rdx+1:3, 1:rdy:3] / 3

    ret_img[1:rdx:3, 0:rdy+1:3,] = img[:x, :y+1]
    ret_img[1:rdx:3, 0:rdy+1:3,] += img[:x, :y+1]
    ret_img[1:rdx:3, 0:rdy+1:3,] += img[1:x+1, :y+1]
    ret_img[1:rdx:3, 0:rdy+1:3,] = ret_img[1:rdx:3, 0:rdy+1:3,] / 3

    #case 2
    ret_img[0:rdx+1:3, 2:rdy:3] = img[:x+1, :y] 
    ret_img[0:rdx+1:3, 2:rdy:3] += img[:x+1, 1:y+1] 
    ret_img[0:rdx+1:3, 2:rdy:3] += img[:x+1, 1:y+1] 
    ret_img[0:rdx+1:3, 2:rdy:3] = ret_img[0:rdx+1:3, 2:rdy:3] / 3

    ret_img[2:rdx:3, 0:rdy+1:3,] = img[:x, :y+1]
    ret_img[2:rdx:3, 0:rdy+1:3,] += img[1:x+1, :y+1]
    ret_img[2:rdx:3, 0:rdy+1:3,] += img[1:x+1, :y+1]
    ret_img[2:rdx:3, 0:rdy+1:3,] = ret_img[2:rdx:3, 0:rdy+1:3,] / 3

    #case 3
    ret_img[1:rdx+1:3, 1:rdy:3] = np.multiply(2, ret_img[1:rdx+1:3, 0:rdy:3])
    ret_img[1:rdx+1:3, 1:rdy:3] += ret_img[1:rdx+1:3, 3:rdy+1:3]
    #ret_img[1:rdx+1:3, 1:rdy:3] += np.ret_img[0:rdx:3, 1:rdy+1:3]
    #ret_img[1:rdx+1:3, 1:rdy:3] += ret_img[0:rdx:3, 1:rdy+1:3]
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
    print(ret_img.shape)
    print(ret_img)
    return ret_img.astype(np.uint8)


if __name__ == '__main__':
    #img = np.array([[1,10,100, 10, 1],[0.1, 0.01, 0.05, 0.01, 0.1], [1, 2, 3, 2, 1],[0.1, 0.01, 0.05, 0.01, 0.1], [1, 10, 100, 10, 1]])
    #upsample_img = bilinear_upsampling(img, (11,11))

    np.set_printoptions(suppress = True)
    np.set_printoptions(precision = 3)   
    img = np.array([[1,10,100, 10],[0.1, 0.01, 0.05, 0.01], [1, 2, 3, 2],[0.1, 0.01, 0.05, 0.01]])
    print(img)
    for x in range(8,10):
        for y in range(8,10):

            upsample_img = bilinear_upsampling(img, (x, y))
    #cv2.imshow('Test', upsample_img)
    #cv2.waitKey(0)
