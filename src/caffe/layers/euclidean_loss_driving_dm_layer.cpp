#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossDrivingDMLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossDrivingDMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num()*bottom[0]->channels();   // batch size
  int dim = count/num;   // equals to number of outputs in last layer
  Dtype y_array[count];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  for (int i = 0; i < num; ++i) {

    if (label[i * dim + 0]>=0)
        y_array[i * dim + 0] = pow(label[i * dim + 0], 1.0/3)*0.4+0.5;     // steering range ~ [-1, 1]
    else
        y_array[i * dim + 0] = -pow(-label[i * dim + 0], 1.0/3)*0.4+0.5;     // steering range ~ [-1, 1]

    //y_array[i * dim + 1] = label[i * dim + 1]*0.4+0.5;   // throttle range ~ [-1, 1]
  }

  caffe_sub(count, bottom_data, y_array, diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;


  //for (int i = 0; i < num; ++i) {
      int i=25;
      for (int j = 0; j < dim; ++j) {
          printf("num: %d, dim: %d, out: %f, y_array: %f, diff: %f \n", i, j, bottom_data[i*dim+j], y_array[i*dim+j], diff_.cpu_data()[i*dim+j]); 
          fflush(stdout);
      }
  //}

}

template <typename Dtype>
void EuclideanLossDrivingDMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (bottom[i]->num()*bottom[i]->channels());
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossDrivingDMLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossDrivingDMLayer);
REGISTER_LAYER_CLASS(EuclideanLossDrivingDM);

}  // namespace caffe
