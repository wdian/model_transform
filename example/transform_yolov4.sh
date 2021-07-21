#!/bin/bash
OldPath=$PWD
cd ./ && rm ./model_output/*

cd ../darknet_2_pb ||exit

python3  darknet_2_h5.py -c "../data/darknet_model/yolov4.cfg"  -w "../data/darknet_model/yolov4.weights"

python3 h5_to_pb.py 'yolov4.pb'

cd "$OldPath" ||exit

