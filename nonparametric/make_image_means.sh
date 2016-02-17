#!/usr/bin/env sh
# Compute the mean image from the TORCS train, validation and test sets.

DB_train=nonparametric/data/TORCS_smooth_np_train
DB_valid=nonparametric/data/TORCS_smooth_np_valid
DB_test=nonparametric/data/TORCS_smooth_np_test

DATA=nonparametric/data
TOOLS=build/tools

$TOOLS/compute_image_mean -backend leveldb \
    $DB_train $DATA/TORCS_smooth_np_train_mean.binaryproto
echo "Done computing training image mean."
$TOOLS/compute_image_mean -backend leveldb \
    $DB_train $DATA/TORCS_smooth_np_valid_mean.binaryproto
echo "Done computing validation image mean."
$TOOLS/compute_image_mean -backend leveldb \
    $DB_train $DATA/TORCS_smooth_np_test_mean.binaryproto
echo "Done computing testing image mean."
