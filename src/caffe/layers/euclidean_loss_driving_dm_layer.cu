#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossDrivingDMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num()*bottom[0]->channels();;   // batch size
  int dim = count/num;   // equals to number of outputs in last layer

  Dtype y_array[count];
  Dtype label[num*dim];
  Dtype bottom_data[count]; 
  Dtype diff[count]; 

  Dtype* y_array_cuda;
  cudaMalloc((void**)&y_array_cuda,sizeof(Dtype)*count);

  const Dtype* bottom_data_cuda = bottom[0]->gpu_data();
  const Dtype* label_cuda = bottom[1]->gpu_data();

  cudaMemcpy(bottom_data,bottom_data_cuda,sizeof(Dtype)*count,cudaMemcpyDeviceToHost);
  cudaMemcpy(label,label_cuda,sizeof(Dtype)*num*dim,cudaMemcpyDeviceToHost);


  for (int i = 0; i < num; ++i) {

    if (label[i * dim + 0]>=0)
        y_array[i * dim + 0] = pow(label[i * dim + 0], 1.0/3)*0.4+0.5;     // steering range ~ [-1, 1]
    else
        y_array[i * dim + 0] = -pow(-label[i * dim + 0], 1.0/3)*0.4+0.5;     // steering range ~ [-1, 1]

    //y_array[i * dim + 1] = label[i * dim + 1]*0.4+0.5;   // throttle range ~ [-1, 1]
  }

  cudaMemcpy(y_array_cuda,y_array,sizeof(Dtype)*count,cudaMemcpyHostToDevice);

  caffe_gpu_sub(count, bottom_data_cuda, y_array_cuda, diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  cudaMemcpy(diff,diff_.gpu_data(),sizeof(Dtype)*count,cudaMemcpyDeviceToHost);
  cudaFree(y_array_cuda);

  //for (int i = 0; i < num; ++i) {
      int i=25;
      for (int j = 0; j < dim; ++j) {
          printf("num: %d, dim: %d, out: %f, y_array: %f, diff: %f \n", i, j, bottom_data[i*dim+j], y_array[i*dim+j], diff[i*dim+j]); 
          fflush(stdout);
      }
  //}

}

template <typename Dtype>
void EuclideanLossDrivingDMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (bottom[i]->num()*bottom[i]->channels());
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossDrivingDMLayer);

}  // namespace caffe
