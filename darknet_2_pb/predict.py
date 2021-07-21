# coding=utf-8
"""
############################
yolov4 pb draw box
############################
"""

import os
import shutil
import cv2
import math
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.python.platform import gfile
from layers.detection_pb import YOLOV4
from utils.util import generate_clors


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

INPUT_NODE = ["input_1:0"]
OUTPUT_NODES = ["output_1:0", "output_2:0", "output_3:0"]


def evaluate(args):
    graph = tf.Graph()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
        with gfile.FastGFile(args.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            sess.run(tf.global_variables_initializer())
            inputs = sess.graph.get_tensor_by_name(INPUT_NODE[0])
            output1 = sess.graph.get_tensor_by_name(OUTPUT_NODES[0])
            output2 = sess.graph.get_tensor_by_name(OUTPUT_NODES[1])
            output3 = sess.graph.get_tensor_by_name(OUTPUT_NODES[2])
            # output = [output1, output2, output3]
            output = [output3, output2, output1]

            predict = YOLOV4(
                output_tensor_list=output,
                model_path=args.model_path,
                anchors_path=args.anchors_path,
                classes_path=args.classes_path,
                image_letterbox_enable=False,
                score=args.detection_thresh,
                iou=args.detection_iou
            )
            colors = generate_clors(predict.class_names)
            index = 0
            with open(os.path.join(args.result_path, "result.txt"), 'w') as fsave:
                image_list = sorted(os.listdir(args.image_path))
                for image_name in image_list:
                    name = os.path.splitext(image_name)[0]
                    img = cv2.imread(os.path.join(args.image_path, image_name), flags=cv2.IMREAD_COLOR)
                    img_detect = img[..., [2, 1, 0]]  # BGR2RGB
                    # input_path = str(Path(args.image_path).parent / "inputs")
                    # if not os.path.exists(input_path):
                    #     input_path = None
                    # bboxes_pr, new_outputs = predict.detect_image(img_detect, inputs, sess)
                    bboxes_pr, new_outputs = predict.detect_image(img_detect, inputs, sess, letterbox=True,
                                                                  outputs=output)

                    if new_outputs is not None:
                        outputs_path = Path(args.result_path) / "outputs"
                        if not outputs_path.exists():
                            os.makedirs(str(outputs_path))
                        with open(str(outputs_path / (name + "_1.bin")), 'wb') as f:
                            f.write(new_outputs[0].astype(np.float32))
                        with open(str(outputs_path / (name + "_2.bin")), 'wb') as f:
                            f.write(new_outputs[1].astype(np.float32))
                        with open(str(outputs_path / (name + "_3.bin")), 'wb') as f:
                            f.write(new_outputs[2].astype(np.float32))
                    print(image_name)
                    for bbox in bboxes_pr:
                        pred_classes_name = predict.class_names[bbox[5]]
                        left = int(max(0, math.floor(bbox[0] + 0.5)))
                        top = int(max(0, math.floor(bbox[1] + 0.5)))
                        right = int(min(img.shape[1], math.floor(bbox[2] + 0.5)))
                        bottom = int(min(img.shape[0], math.floor(bbox[3] + 0.5)))
                        _res = [name, pred_classes_name, "%f" % bbox[4], "%f" % left, "%f" % top, "%f" % right, "%f" % bottom]
                        if index > 0:
                            fsave.write('\n')
                        fsave.write("%s %s %f %f %f %f %f" % (name, pred_classes_name, bbox[4], left, top, right, bottom))
                        cv2.rectangle(img, (left, top), (right, bottom), colors[bbox[5]], 2)
                        index = index + 1
                    cv2.imwrite(os.path.join(args.result_path, image_name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate classification or detection performance')
    parser.add_argument('-image_path', help='image path',
                        default='../data/images/')
    parser.add_argument('-result_path', help='detection result save path',
                        default='../data/result/')
    parser.add_argument('-model_path', help='mode path',
                        default='../model_output/yolov4_zhiye_v7.12.pb')

    parser.add_argument('-classes_path',  help="classes path", default='../data/source/coco_classes.txt')
    parser.add_argument('-anchors_path',  help="anchors path", default='../data/source/yolov4_anchors.txt')
    parser.add_argument('-resize_type', default='letterbox', help="image resize type ['letterbox', 'normal']")
    parser.add_argument('-detection_iou', type=float, default=0.5,
                        help="Threshold of IOU ratio to determine a match bbox.")
    parser.add_argument('-detection_thresh', type=float, default=0.2,  help="""Threshold of confidence score for 
    calculating evaluation metric, default 0.2.""")

    args = parser.parse_args()
    evaluate(args)
