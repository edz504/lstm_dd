#!/usr/bin/env sh

TOOLS=../build/tools

#GLOG_logtostderr=1 $TOOLS/torcs_benchmark.bin pre_trained/driving_run_1F.prototxt pre_trained/driving_chenyi_iter_140000.caffemodel GPU

#GLOG_logtostderr=1 $TOOLS/torcs_benchmark.bin 1F_01/driving_run_1F.prototxt 1F_01/driving_train_1F_iter_20000.caffemodel GPU

GLOG_logtostderr=1 $TOOLS/torcs_benchmark.bin lstm_11/driving_run_lstm.prototxt lstm_11/driving_train_lstm_all_iter_20000.caffemodel GPU

#GLOG_logtostderr=1 $TOOLS/torcs_benchmark_draw.bin
