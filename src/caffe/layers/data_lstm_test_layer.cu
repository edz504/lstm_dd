#include <stdint.h>
#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

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

extern IplImage* leveldbTest;

template <typename Dtype>
void DataLstmTestLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

INSTANTIATE_LAYER_GPU_FUNCS(DataLstmTestLayer);

}  // namespace caffe
