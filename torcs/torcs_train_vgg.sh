#!/usr/bin/env sh

TOOLS=../build/tools

#$TOOLS/caffe train \
#  --solver=vgg16_2/driving_solver_stage1.prototxt \
#  --snapshot=bvlc_chenyi_vgg16net.caffemodel

$TOOLS/caffe train \
  --solver=vgg16_2/driving_solver_stage2.prototxt \
  --snapshot=vgg16_2/driving_train_1F_iter_40000.solverstate


