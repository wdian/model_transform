# -*- coding: utf-8 -*-#
"""
-------------------------------------------------------------------
   Copyright (c) 2019-2022 Snow Lake Inc. All rights reserved.

   Description :
   File Name：     __init__.py.py
   Author :       wangdian@snowlake-tech.com
   create date：   2021/7/8
-------------------------------------------------------------------
"""
import cv2
import numpy as np
import numba


@numba.jit(nopython=True)
def resize_image(image, w, h):
    """
    resize image
    Args:
        image: h,w,c, normal
        w: width
        h: height

    Returns:
        h*w*c
    """
    resized = np.zeros((h, w, image.shape[2]), dtype=np.float32)
    part = np.zeros((image.shape[0], w, image.shape[2]), dtype=np.float32)
    w_scale = (image.shape[1] - 1.0) / (resized.shape[1] - 1.0)
    h_scale = (image.shape[0] - 1.0) / (resized.shape[0] - 1.0)
    for k in range(image.shape[2]):
        for r in range(image.shape[0]):
            for c in range(w):
                if c == w - 1 or image.shape[1] == 1:
                    val = image[r, image.shape[1] - 1, k]
                else:
                    sx = c * w_scale
                    ix = int(sx)
                    dx = sx - ix
                    val = (1 - dx) * image[r, ix, k] + dx * image[r, ix + 1, k]
                part[r, c, k] = val

    for k in range(image.shape[2]):
        for r in range(h):
            sy = r * h_scale
            iy = int(sy)
            dy = sy - iy
            for c in range(w):
                val = (1 - dy) * part[iy, c, k]
                resized[r, c, k] = val

            if r == h - 1 or image.shape[0] == 1:
                continue
            for c in range(w):
                val = dy * part[iy + 1, c, k]
                resized[r, c, k] += val

    return resized


def letterbox_image(image, new_size, padding_value=0.5):
    """
    resize image with unchanged aspect ratio using padding
    Args:
        image: h,w,c, normal
        new_size:w,h
        padding_value: padding value

    Returns:
        h,w,c
    """
    ih = image.shape[0]
    iw = image.shape[1]
    w, h = new_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), cv2.INTER_LINEAR)
    offset = ((w - nw) // 2, (h - nh) // 2)
    new_image = np.full((new_size[1], new_size[0], 3), padding_value, dtype=np.uint8)
    new_image[offset[1]:offset[1] + nh, offset[0]:offset[0] + nw, :] = image

    return new_image


def letterbox_image_resize(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    h, w, _ = image.shape
    if image.shape[0] != 416 or image.shape[1] != 416:
        print('images resize:', w, h)
        new_image = cv2.resize(image, (size[0], size[1]), cv2.INTER_LINEAR)
    else:
        new_image = image
    return new_image


def _resize_image_test_(image_path, test_data_path, model_image_size):
    image_data2 = np.fromfile(test_data_path, dtype=np.float32).reshape((3, model_image_size[0], model_image_size[1]))
    image_data4 = np.swapaxes(image_data2, 0, 1)
    image_data4 = np.swapaxes(image_data4, 1, 2)

    # image_data = np.fromfile(image_path, dtype=np.float32).reshape((3, 1426, 1920))
    # image_data_src = np.zeros((1426, 1920, 3))
    # image_data_src[:, :, 0] = image_data[0, :, :]
    # image_data_src[:, :, 1] = image_data[1, :, :]
    # image_data_src[:, :, 2] = image_data[2, :, :]

    img = np.array(cv2.imread(image_path, flags=cv2.IMREAD_COLOR), dtype='float32')
    img /= 255.
    img = img[..., [2, 1, 0]]
    resized_data = resize_image(img, model_image_size[0], model_image_size[1])
    resized_data = resized_data - image_data4
    print(resized_data.shape)
    print(resized_data[resized_data > 0.001].shape)
    print(resized_data[resized_data < -0.001].shape)
    print("test done.")


def _resize_image_test_letterbox_(image_path, model_image_size):
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img2 = letterbox_image(img, model_image_size)

    cv2.imshow("fff", img2)
    cv2.waitKey(1000000)
    # print(resized_data.shape)
    # print(resized_data[resized_data > 0.001].shape)
    # print(resized_data[resized_data < -0.001].shape)
    print("test done.")


