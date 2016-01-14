#!/usr/bin/env sh

TOOLS=../build/tools

# 2 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_2lane.bin lstm/driving_run_lstm.prototxt lstm/driving_train_lstm_iter_140000.caffemodel GPU

# 3 lane
GLOG_logtostderr=1 $TOOLS/torcs_run_3lane.bin lstm_11/driving_run_lstm.prototxt lstm_11/driving_train_lstm_iter_20000.caffemodel GPU

# 1 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_1lane.bin lstm/driving_run_lstm.prototxt lstm/driving_train_lstm_iter_140000.caffemodel GPU

