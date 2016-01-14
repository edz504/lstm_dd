#!/usr/bin/env sh

../build/tools/caffe train \
    --solver=lstm_12/driving_solver_lstm.prototxt \
    --snapshot=pre_trained/driving_chenyi_iter_140000.caffemodel



