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
    # | 1 | 3 | 4 | 1 | 
    # -----------------
    # | 2 | 5 | 6 | 2 |   
    # -----------------
    # | d | 1 | 2 | e |   
    # -----------------

    ret_img[0:rdx+1:3, 0:rdy+1:3] = img[:x+1, :y+1]

    # case 1
    ret_img[0:rdx+1:3, 1:rdy:3] = np.multiply(img[:x+1, :y], 2)
    ret_img[0:rdx+1:3, 1:rdy:3] += img[:x+1, 1:y+1] 
    ret_img[0:rdx+1:3, 1:rdy:3] = ret_img[0:rdx+1:3, 1:rdy:3] / 3

    ret_img[1:rdx:3, 0:rdy+1:3,] = np.multiply(img[:x, :y+1], 2)
    ret_img[1:rdx:3, 0:rdy+1:3,] += img[1:x+1, :y+1]
    ret_img[1:rdx:3, 0:rdy+1:3,] = ret_img[1:rdx:3, 0:rdy+1:3,] / 3

    #case 2
    ret_img[0:rdx+1:3, 2:rdy:3] = img[:x+1, :y] 
    ret_img[0:rdx+1:3, 2:rdy:3] += np.multiply(img[:x+1, 1:y+1], 2)
    ret_img[0:rdx+1:3, 2:rdy:3] = ret_img[0:rdx+1:3, 2:rdy:3] / 3

    ret_img[2:rdx:3, 0:rdy+1:3] = img[:x, :y+1]
    ret_img[2:rdx:3, 0:rdy+1:3] += np.multiply(img[1:x+1, :y+1], 2)
    ret_img[2:rdx:3, 0:rdy+1:3] = ret_img[2:rdx:3, 0:rdy+1:3,] / 3

    #case 3
    ret_img[1:rdx+1:3, 1:rdy:3] =  np.multiply(ret_img[1:rdx+1:3, 0:rdy:3], 2)   #left
    ret_img[1:rdx+1:3, 1:rdy:3] += ret_img[1:rdx+1:3, 3:rdy+1:3]                 #right
    ret_img[1:rdx+1:3, 1:rdy:3] += np.multiply(ret_img[0:rdx:3, 1:rdy+1:3], 2)   #top
    ret_img[1:rdx+1:3, 1:rdy:3] += ret_img[3:rdx+1:3, 1:rdy+1:3]                 #down
    ret_img[1:rdx+1:3, 1:rdy:3] =  ret_img[1:rdx+1:3, 1:rdy:3] / 6

    #case 4
    ret_img[1:rdx+1:3, 2:rdy:3] =  ret_img[1:rdx+1:3, 0:rdy:3]                   #left
    ret_img[1:rdx+1:3, 2:rdy:3] += np.multiply(ret_img[1:rdx+1:3, 3:rdy+1:3], 2) #right
    ret_img[1:rdx+1:3, 2:rdy:3] += np.multiply(ret_img[0:rdx:3, 2:rdy+1:3], 2)   #top
    ret_img[1:rdx+1:3, 2:rdy:3] += ret_img[3:rdx+1:3, 2:rdy+1:3]                 #down
    ret_img[1:rdx+1:3, 2:rdy:3] =  ret_img[1:rdx+1:3, 2:rdy:3] / 6

    #case 5
    ret_img[2:rdx+1:3, 1:rdy:3] =  np.multiply(ret_img[2:rdx+1:3, 0:rdy:3], 2)   #left
    ret_img[2:rdx+1:3, 1:rdy:3] += ret_img[2:rdx+1:3, 3:rdy+1:3]                 #right
    ret_img[2:rdx+1:3, 1:rdy:3] += ret_img[0:rdx:3, 1:rdy+1:3]                   #top
    ret_img[2:rdx+1:3, 1:rdy:3] += np.multiply(ret_img[3:rdx+1:3, 1:rdy+1:3], 2) #down
    ret_img[2:rdx+1:3, 1:rdy:3] =  ret_img[2:rdx+1:3, 1:rdy:3] / 6

    #case 6
    ret_img[2:rdx+1:3, 2:rdy:3] =  ret_img[2:rdx+1:3, 0:rdy:3]   				 #left
    ret_img[2:rdx+1:3, 2:rdy:3] += np.multiply(ret_img[2:rdx+1:3, 3:rdy+1:3], 2) #right
    ret_img[2:rdx+1:3, 2:rdy:3] += ret_img[0:rdx:3, 2:rdy+1:3]                   #top
    ret_img[2:rdx+1:3, 2:rdy:3] += np.multiply(ret_img[3:rdx+1:3, 2:rdy+1:3], 2) #down
    ret_img[2:rdx+1:3, 1:rdy:3] =  ret_img[2:rdx+1:3, 1:rdy:3] / 6

    #Case right up
    # -------------
    # | b | 3 | c |
    # -------------
    # | 1 | 3 | 1 |
    # -------------
    # | 2 | 3 | 2 |
    # -------------
    # | e | 3 | f |
    # -------------

    ret_img[:rdx+1:3, rdy+2::2] = img[:x+1, y+1:]

    #case 1
    ret_img[1:rdx+1:3, rdy+2::2] =  np.multiply(ret_img[0:rdx:3, rdy+2::2], 2) 	#top
    ret_img[1:rdx+1:3, rdy+2::2] += ret_img[3:rdx+1:3, rdy+2::2]			   	#down
    ret_img[1:rdx+1:3, rdy+2::2] =  ret_img[1:rdx+1:3, rdy+2::2] / 3

    #case 2
    ret_img[2:rdx+1:3, rdy+2::2] =  ret_img[0:rdx:3, rdy+2::2]					#top
    ret_img[2:rdx+1:3, rdy+2::2] += np.multiply(ret_img[3:rdx+1:3, rdy+2::2], 2)#down
    ret_img[2:rdx+1:3, rdy+2::2] =  ret_img[2:rdx+1:3, rdy+2::2] / 3

    #case3
    ret_img[:rdx, rdy+1::2] =  ret_img[:rdx, rdy:-1:2] 		#left
    ret_img[:rdx, rdy+1::2] += ret_img[:rdx, rdy+2::2] 		#right
    ret_img[:rdx, rdy+1::2] =  ret_img[:rdx, rdy+1::2] / 2

    #Case left down
    # -----------------
    # | d | 1 | 2 | e |
    # -----------------
    # | 3 | 3 | 3 | 3 |
    # -----------------
    # | g | 1 | 2 | h |
    # -----------------

    ret_img[rdx+2::2, :rdy+1:3]  =  img[x+1:, :y+1]
    #case 1
    ret_img[rdx+2::2, 1:rdy+1:3] =  np.multiply(ret_img[rdx+2::2, :rdy:3], 2)   #left
    ret_img[rdx+2::2, 1:rdy+1:3] += ret_img[rdx+2::2, 3:rdy+1:3]			   	#right
    ret_img[rdx+2::2, 1:rdy+1:3] =  ret_img[rdx+2::2, 1:rdy+1:3] / 3

    #case 2
    ret_img[rdx+2::2, 2:rdy+1:3] =  ret_img[rdx+2::2, :rdy:3]					#left
    ret_img[rdx+2::2, 2:rdy+1:3] += np.multiply(ret_img[rdx+2::2, 3:rdy+1:3], 2)#right
    ret_img[rdx+2::2, 2:rdy+1:3] =  ret_img[rdx+2::2, 2:rdy+1:3] / 3

    #case3
    ret_img[rdx+1::2, :rdy+1:] =  ret_img[rdx:-1:2, :rdy+1:] 					#top
    ret_img[rdx+1::2, :rdy+1:] += ret_img[rdx+2::2, :rdy+1:] 					#down
    ret_img[rdx+1::2, :rdy+1:] =  ret_img[rdx+1::2, :rdy+1:] / 2
    
    return ret_img.astype(np.uint8)


if __name__ == '__main__':

    '''
    np.set_printoptions(suppress = True)
    np.set_printoptions(precision = 2)   
    img = np.array([[1,10,50, 10, 1],[0.1, 0.01, 0.5, 0.01, 0.1], [1, 2, 3, 2, 1],[0.1, 0.01, 0.5, 0.01, 0.1], [1, 10, 50, 10, 1]])
    #img = np.array([[1,10,100, 10],[0.1, 0.01, 0.05, 0.01], [1, 2, 3, 2],[0.1, 0.01, 0.05, 0.01]])
    print(img)
    for x in range(10,12):
        for y in range(10,12):

            upsample_img = bilinear_upsampling(img, (x, y))
    '''
    img = cv2.imread('./task1and2_hybrid_pyramid/1_bicycle.bmp')
    print(img.shape)
    upsample_img = bilinear_upsampling(img,(img.shape[0]*2, img.shape[1]*2 + 1, img.shape[2]))
    print(upsample_img.shape)
    cv2.imshow('Test', upsample_img)
    cv2.waitKey(0)

    #cv2.imshow('Test', upsample_img)
    #cv2.waitKey(0)
