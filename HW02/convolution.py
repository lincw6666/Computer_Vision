import numpy as np
from cv2 import cv2
import sys
import time


def _is_image_type_valid(img):
    assert isinstance(img, np.ndarray), '[_is_image_type_valid] Error!!' +\
        'Invalid image type: '+str(type(img))+'. It should be a np.ndarray.'
    assert len(img.shape) == 3, '[_is_image_type_valid] Error!! An image ' +\
        'should have 3 dimension. But we get ' + len(img.shape) + '.'


def _image_reshape(img, shape):
    _is_image_type_valid(img)
    h, w, _ = shape
    assert img.shape[0] == h*w, '[_image_reshape] Error!! Unmatch image '+\
        'and reshape image size!!'
    return np.array([img[w*i: w*(i+1), 0, :] for i in range(h)], dtype=np.uint8)


def _array_to_vector(img, kernel_size, padding=None):
    # Set kernel size and image size.
    k_rows, k_cols = kernel_size
    assert isinstance(k_rows, int) and isinstance(k_cols, int)
    rows, cols, channels = img.shape
    assert isinstance(rows, int) and isinstance(cols, int) and\
        isinstance(channels, int)

    # Set padding.
    if padding is not None:
        assert isinstance(padding, tuple) and len(padding) == 2
        padx, pady = padding
    else:
        padx, pady = k_rows//2, k_cols//2

    # Pad the image.
    padded_img_shape = (rows+2*padx, cols+2*pady, channels)
    padded_img = np.zeros(padded_img_shape)
    padded_img[padx:rows+padx, pady:cols+pady, :] = img

    row_channel_len = padded_img_shape[1] * channels

    # The starting point (left top corner) of each convolution.
    start_idx = np.array([
        [j*channels + row_channel_len*i + k\
            for i in range(rows - k_rows + 1 + 2*padx)\
                for j in range(cols - k_cols + 1 + 2*pady)]\
        for k in range(channels)])
    # Every accessed elements of each convolution.
    grid = np.array(
        np.tile([j*channels+(row_channel_len)*i\
            for i in range(k_rows) for j in range(k_cols)], (channels, 1)))
    to_take = start_idx[:, :, None] + grid[:, None, :]

    return [padded_img.take(to_take[i]) for i in range(channels)]


def convolution(img, kernel, padding=None):
    # Check the kernel's properties.
    assert isinstance(kernel, np.ndarray), '[convolution] Error!! Invalid' +\
        'kernel type: ' + str(type(kernel)) + '. It should be a np.ndarray.'
    
    # Main task.
    vectorized_img = _array_to_vector(img, kernel.shape, padding=padding)
    kernel = kernel.reshape(-1, 1)
    return _image_reshape(
        np.stack(
            [np.matmul(vec_img, kernel) for vec_img in vectorized_img], axis=-1
        ),
        img.shape)


def debug():
    img = cv2.imread('hw2_data/task1and2_hybrid_pyramid/1_bicycle.bmp')
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    kernel = kernel / kernel.sum()
    processed_img = convolution(img, kernel)
    cv2.imshow('test', processed_img)
    cv2.waitKey(0)
