# -*- coding:utf-8 -*- #
"""
-------------------------------------------------------------------
   Copyright (c) 2019-2022 Snow Lake Inc. All rights reserved.

   Description :
   File Name：     util.py
   Author :       wangdian@snowlake-tech.com
   create date：   2021/7/9
-------------------------------------------------------------------
"""
import colorsys
import numpy as np


def generate_clors(class_names):
    num_classes = len(class_names)
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(23)
    np.random.shuffle(colors)
    np.random.seed(None)

    return colors
