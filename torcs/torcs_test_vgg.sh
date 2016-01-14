#!/usr/bin/env sh

TOOLS=../build/tools
FOLDER=vgg16_2

# 2 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_2lane.bin $FOLDER/driving_run_1F.prototxt $FOLDER/driving_train_1F_iter_40000.caffemodel GPU

# 3 lane
GLOG_logtostderr=1 $TOOLS/torcs_run_3lane.bin $FOLDER/driving_run_1F.prototxt $FOLDER/driving_train_1F_iter_40000.caffemodel GPU

# 1 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_1lane.bin $FOLDER/driving_run_1F.prototxt $FOLDER/driving_train_1F_iter_40000.caffemodel GPU

