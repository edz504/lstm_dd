#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoidHist(Dtype x) {
  return 1. / (1. + exp(-x));
}

/////////////////// by chenyi
extern vector<float> output_blob;
/////////////////// by chenyi

template <typename Dtype>
void SigmoidHistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoidHist(bottom_data[i]);
  }

/////////////////// by chenyi
  const Dtype* top_blob = top[0]->cpu_data();
  for (int i=0;i<count;i++) {
      output_blob[i]=top_blob[i];
  }
/////////////////// by chenyi
}

template <typename Dtype>
void SigmoidHistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidHistLayer);
#endif

INSTANTIATE_CLASS(SigmoidHistLayer);


}  // namespace caffe
