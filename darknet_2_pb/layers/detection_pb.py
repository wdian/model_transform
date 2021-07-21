# -*- coding:utf-8 -*- #
# @Time    : 2019/6/27 下午4:27
# @Author  : MBF
# @FileName: detection_try.py.py
# @Software: PyCharm
from __future__ import absolute_import, division, print_function, unicode_literals
from timeit import default_timer as timer

import cv2
from tensorflow.python.platform import gfile
import tensorflow as tf
import colorsys
import os
import numpy as np
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import utils.image_resize


class YOLOV4(object):
    def __init__(self, output_tensor_list, model_path, anchors_path, classes_path, image_letterbox_enable=True, score=0.25, iou=0.5, model_image_size=(416, 416)):

        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.image_letterbox_enable = image_letterbox_enable

        self.score = score
        self.iou = iou

        self.model_image_size = model_image_size

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.get_default_session()
        self.output_tensor_list = output_tensor_list
        self.input_image_shape = K.placeholder(shape=(2,))
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # load pb
        self.boxes, self.scores, self.classes = self.generate(self.output_tensor_list)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def yolo_head(self, feats, anchors, num_classes, input_shape, xyscale, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # reshape ->(1,1,1,2,2)
        grid_shape = K.shape(feats)[1:3]  # height, width (19,19) (?,19,19,255)  -> (19,19)

        # 生成网格矩阵为网格建立索引
        # grid_y和grid_x用于生成网格grid，通过arange、reshape、tile的组合， 创建y轴的0~19的组合grid_y，再创建x轴的0~19的组合grid_x，将两者拼接concatenate，就是grid；
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        print('grid_shape:', grid_x.shape, grid_y.shape)
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, tf.float32)
        print('grid_shape', grid.shape)

        # 255 = 3 × (80 + 5)（x, y ,w, h ,confidence） 19,19,3,85
        # 将feats的最后一维展开，将anchors与其他数据（类别数+4个框值+框置信度）分离
        feats = K.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
        # x, y
        box_xy = ((K.sigmoid(feats[..., :2]) * xyscale) - 0.5 * (xyscale - 1) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
        # 获取准确度的分数
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_enable=True):
        """
        Get corrected boxes
        :param box_xy:
        :param box_wh:
        :param input_shape:
        :param image_shape:
        :return:
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        # print('第几个网格里的相对偏移：', box_hw.shape)
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        # 将张量中的元素四舍五入成为最接近的整数x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5]) tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
        if letterbox_enable:
            new_shape = tf.round(image_shape * K.min(input_shape / image_shape))
        else:
            new_shape = input_shape
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape  # 缩放比例
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])
        # # (3) clip some boxes those are out of range

        # # (4) discard some invalid boxes

        # Scale boxes back to original image shape.
        a = K.concatenate([image_shape, image_shape])
        print('------', boxes.shape, K.concatenate([image_shape, image_shape]).shape)
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes

    def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape, xyscale):
        '''Process Conv layer output'''
        # box_xy是box的中心坐标，(0~1) 相对位置；box_wh是box的宽高，(0~1)相对值；
        # box_confidence是框中物体置信度；box_class_probs是类别置信度；
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats,
                                                                         anchors, num_classes, input_shape, xyscale)
        # 将box_xy和box_wh的(0~1)相对值，转换为真实坐标，输出boxes是(y_min,x_min,y_max,x_max)的值
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_enable=self.image_letterbox_enable)

        # reshape,将不同网格的值转换为框的列表。即（?,13                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ,13,3,4）->(?,4)  ？：框的数目
        boxes = K.reshape(boxes, [-1, 4])
        # 框的得分=框的置信度*类别置信度
        box_scores = box_confidence * box_class_probs
        # reshape,将框的得分展平，变为(?,80); ?:框的数目
        box_scores = K.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    def yolo_eval(self, yolo_outputs,
                  anchors,
                  num_classes,
                  image_shape,
                  max_boxes=2000,
                  score_threshold=0.001,
                  iou_threshold=.45):
        """Evaluate YOLO model on given input and return filtered boxes."""
        print('shape:', K.shape(yolo_outputs[0])[1:3])
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if len(yolo_outputs) == 3 else [[1, 2, 3],[3, 4, 5]]
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        print('input_shape', input_shape)
        boxes = []
        box_scores = []
        XYSCALES = [1.05, 1.1, 1.2]
        for l in range(num_layers):
            xyscale = XYSCALES[l]
            _boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l],
                                                             anchors[anchor_mask[l]], num_classes, input_shape,
                                                             image_shape, xyscale)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        boxes = K.concatenate(boxes, axis=0)  # K.concatenate:将数据展平 ->(?,4)
        box_scores = K.concatenate(box_scores, axis=0)
        # score_threshold = 0.25
        # MASK掩码，过滤小于score阈值的值，只保留大于阈值的值
        mask = box_scores >= score_threshold
        # 最大检测框数20
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        # iou_threshold = 0.45
        for c in range(num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])  # 通过掩码MASK和类别C筛选框boxes
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])  # 通过掩码MASK和类别C筛选scores
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)  # K.gather:根据索引nms_index选择class_boxes
            class_box_scores = K.gather(class_box_scores, nms_index)  # 根据索引nms_index选择class_box_score)
            classes = K.ones_like(class_box_scores, 'int32') * c  # 计算类的框得分
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_,
                               axis=0)  # K.concatenate().将相同维度的数据连接在一起；把boxes_展平。  -> 变成格式:(?,4);  ?:框的个数；4：（x,y,w,h）
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    @staticmethod
    def resize_image(image, new_size, letterbox):
        """

        Args:
            image: RGB, h,w,c
            new_size: resize size
            letterbox: bool

        Returns:

        """
        image = image / 255
        if letterbox:
            # return utils.image_resize.letterbox_image(image, new_size)
            # 直接缩放
            return utils.image_resize.letterbox_image_resize(image, new_size)

        return utils.image_resize.resize_image(image, new_size[0], new_size[1])

    def generate(self, outputs):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.pb'), 'tensorflow model or weights must be a .pb file.'

        boxes, scores, classes = self.yolo_eval(outputs, self.anchors,
                                                len(self.class_names), self.input_image_shape,
                                                score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def __read_resized_image_data__(self, data_path):
        image_data2 = np.fromfile(data_path, dtype=np.float32).reshape((3, self.model_image_size[0], self.model_image_size[1]))
        image_data4 = np.zeros((self.model_image_size[0], self.model_image_size[1], 3))
        image_data4[:, :, 0] = image_data2[0, :, :]
        image_data4[:, :, 1] = image_data2[1, :, :]
        image_data4[:, :, 2] = image_data2[2, :, :]
        return image_data4

    def detect_image(self, image, inputs, sess, letterbox=False, outputs=None, input_path=None):
        """

        Args:
            image: cv2.imread(patn, flags=cv2.IMREAD_COLOR), to RGB
            inputs:
            sess:
            letterbox:
            outputs:
            input_path:

        Returns:

        """
        if input_path is not None:
            image_data = self.__read_resized_image_data__(input_path)
        else:
            if self.model_image_size != (None, None):
                assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
                new_image_size = self.model_image_size
            else:
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
            boxed_image = self.resize_image(image, new_image_size, letterbox)
            image_data = np.array(boxed_image, dtype='float32')
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        new_outputs = None
        if outputs is not None:
            new_outputs = sess.run(outputs, feed_dict={'input_1:0': image_data})

        out_boxes, out_scores, out_classes = sess.run([self.boxes, self.scores, self.classes], feed_dict={
            self.input_image_shape: [image.shape[0], image.shape[1]], inputs: image_data})
        result = []

        for i, one_coord in enumerate(out_boxes):
            one_coord = list(one_coord)
            one_coord[0], one_coord[1], one_coord[2], one_coord[3] = one_coord[1], one_coord[0], one_coord[3], \
                                                                     one_coord[2]
            one_coord.append(out_scores[i])
            one_coord.append(out_classes[i])
            result.append(one_coord)
        return result, new_outputs

    @staticmethod
    def close_session():
        K.clear_session()
