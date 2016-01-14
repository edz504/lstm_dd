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

#define para_dim 14     // dimension of affordance

namespace caffe {

template <typename Dtype>
DataHistLayer<Dtype>::~DataHistLayer<Dtype>() {}

vector<float> output_blob;

template <typename Dtype>
void DataHistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> shape(3);
  shape[0] = bottom[0]->num();      /////// bottom[0]->num() must be 1
  shape[1] = bottom[0]->channels();
  shape[2] = para_dim;

  top[0]->Reshape(shape);  

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  int count=bottom[0]->num()*bottom[0]->channels()*para_dim;
  for (int i=0;i<count;i++) 
      output_blob.push_back(0);

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataHistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const int sequence_size = bottom[0]->num();   ///// sequence_size must be 1
  const int ind_seq_num = bottom[0]->channels();
  int item_id;

  for (int time_id = 0; time_id < sequence_size; ++time_id) {
     for (int seq_id = 0; seq_id < ind_seq_num; ++seq_id) {
        item_id=time_id*ind_seq_num+seq_id;

        if (bottom_data[item_id]>0.5) {
            for (int j = 0; j < para_dim; ++j)
                top_data[item_id * para_dim + j] = output_blob[item_id * para_dim + j]; 
        } else {
            for (int j = 0; j < para_dim; ++j)
                top_data[item_id * para_dim + j] = 0;
        }
     }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DataHistLayer, Forward);
#endif

INSTANTIATE_CLASS(DataHistLayer);
REGISTER_LAYER_CLASS(DataHist);

}  // namespace caffe
