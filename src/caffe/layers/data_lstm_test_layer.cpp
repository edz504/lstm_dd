#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//////////////// by chenyi
#include <leveldb/db.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#define resize_width 280
#define resize_height 210
//////////////// by chenyi

namespace caffe {

template <typename Dtype>
DataLstmTestLayer<Dtype>::~DataLstmTestLayer<Dtype>() {}


extern leveldb::DB* db_tmp;
IplImage* leveldbTest=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);

template <typename Dtype>
void DataLstmTestLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db_tmp);

  // Read a data point, and use it to initialize the top blob.
  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  snprintf(key_cstr, kMaxKeyLength, "%08d", 1);
  db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
  Datum datum;
  datum.ParseFromString(value);

  int sequence_size=this->layer_param_.data_lstm_test_param().sequence_size();
  // image 
  top[0]->Reshape(sequence_size, datum.channels(), datum.height(), datum.width());

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // marker
  top[1]->Reshape(sequence_size, 1, 1, 1);

  const string& mean_file = this->layer_param_.data_lstm_test_param().mean_file();
  LOG(INFO) << "Loading mean file from: " << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  data_mean_.FromProto(blob_proto);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLstmTestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Datum datum;
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_marker = top[1]->mutable_cpu_data();

  // datum scales
  const int size = resize_height*resize_width*3;
  const Dtype* mean = this->data_mean_.mutable_cpu_data();

  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  const int sequence_size = this->layer_param_.data_lstm_test_param().sequence_size();

  for (int seq_id = 0; seq_id < sequence_size; ++seq_id) {

      snprintf(key_cstr, kMaxKeyLength, "%08d", seq_id+1);
      db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
      datum.ParseFromString(value);
      const string& data = datum.data();

      for (int j = 0; j < size; ++j) {
         Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[j]));
         top_data[seq_id * size + j] = (datum_element - mean[j]);
      }

      top_marker[seq_id] = 1;        
      //top_marker[seq_id] = datum.float_data(0);   // for lstm bechmark only (used with torcs_bechmark.cpp)

      if (seq_id == 0) {
         for (int h = 0; h < resize_height; ++h) {
            for (int w = 0; w < resize_width; ++w) {
               leveldbTest->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
               leveldbTest->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
               leveldbTest->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
             }
         }
         cvShowImage("Image from leveldb", leveldbTest);
         cvWaitKey( 1 );
      }

  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DataLstmTestLayer, Forward);
#endif

INSTANTIATE_CLASS(DataLstmTestLayer);
REGISTER_LAYER_CLASS(DataLstmTest);

}  // namespace caffe
