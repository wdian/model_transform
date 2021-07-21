# -*- coding: utf-8 -*-

import os
import sys
import os.path as osp

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from layers.mish import Mish

# 路径参数
input_path = '../model_output/'
weight_file = 'yolo.h5'
output_path = '../model_output/'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
weight_file_path = osp.join(input_path, weight_file)
# output_graph_name = weight_file[:-3] + '.pb'
output_graph_name = sys.argv[1]

model_path = osp.join(input_path, output_graph_name)


# 转换函数
def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=False):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util, graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


def generate_tensorboard_graph():
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()

    graph_def.ParseFromString(tf.gfile.FastGFile(model_path, 'rb').read())
    tf.import_graph_def(graph_def, name='yolov4')
    tf.summary.FileWriter('log/', graph)


if __name__ == '__main__':
    # 加载模型
    h5_model = load_model(weight_file_path, custom_objects={'Mish': Mish})
    h5_to_pb(h5_model, output_dir=output_path, model_name=output_graph_name)

    print('model saved')
