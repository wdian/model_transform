# -*- coding:utf-8 -*- #
"""
-------------------------------------------------------------------
   Copyright (c) 2019-2022 Snow Lake Inc. All rights reserved.

   Description :
   File Name：     mish.py
   Author :       wangdian@snowlake-tech.com
   create date：   2021/7/8
-------------------------------------------------------------------
"""
from keras import backend as K
from keras.engine.base_layer import Layer


class Mish(Layer):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        Swish 的表达式为 f(x) = x · sigmoid(x)
        >> X_input = Input(input_shape)
        >> X = Mish()(X_input)
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))
        # return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
