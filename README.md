# model_transform
transform darknet to tensorflow

## Introduction

this code for transform darknet  model to tensorflow model

Supported models include **yolov3_tiny**, **yolov3**, **yolov4**

## Requirements

Keras=2.2.4

tensorflow=1.14.0

numpy =1.19.2

numba=0.53.1

opencv-python=4.5.1.48

## Usage

To use this package, you need a darknet model, include  weights file(\*.weights) and config file(\*.cfg) .Put it in the **data/darknet_model** directory

example:

```shell
cd darknet_2_pb 

python3  darknet_2_h5.py -c "../data/darknet_model/yolov3.cfg"  -w "../data/darknet_model/yolov3.weights"
python3 h5_to_pb.py 'yolov3.pb'
```

you can run 

```shell
./example/transform_yolov3.sh
```

