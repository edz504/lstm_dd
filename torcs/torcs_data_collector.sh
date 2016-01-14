#!/usr/bin/env sh

TOOLS=../build/tools

#GLOG_logtostderr=1 $TOOLS/torcs_data_collector_DM.bin

GLOG_logtostderr=1 $TOOLS/torcs_convert_leveldb_for_lstm.bin
