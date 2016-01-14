#include <stdint.h>
#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

#define para_dim 14     // dimension of affordance

namespace caffe {

extern vector<float> output_blob;

template <typename Dtype>
void DataHistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

INSTANTIATE_LAYER_GPU_FUNCS(DataHistLayer);

}  // namespace caffe
