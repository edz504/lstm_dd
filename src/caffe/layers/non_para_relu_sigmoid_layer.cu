#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernel for forward
template <typename Dtype>
__global__ void NonParaReLUSigmoidForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* beta_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = (in[index] > 0 ?
                  in[index] : 0) + beta_data[c] * 1. / (1. + exp(-in[index]));
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void NonParaReLUSigmoidBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* beta_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + beta_data[c] * 1. / (1. + exp(-in_data[index]))
        * (1 - 1. / (1. + exp(-in_data[index]))));
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void NonParaReLUSigmoidParamBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    out_diff[index] = in_diff[index] * 1. / (1. + exp(-in_data[index]));
  }
}

template <typename Dtype>
void NonParaReLUSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* beta_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  NonParaReLUSigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, beta_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void NonParaReLUSigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* beta_diff = this->blobs_[0]->mutable_gpu_diff();
    // beta_diff is set as 0, then accumulated over batches
    caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), beta_diff);
    int cdim = channels * dim;
    Dtype dsum = 0.;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      Dtype* temp_buff = multiplier_.mutable_gpu_diff();
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      NonParaReLUSigmoidParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n), multiplier_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, multiplier_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            multiplier_.gpu_diff(), multiplier_.gpu_data(), 1.,
            beta_diff);
      }
    }
    if (channel_shared_) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), beta_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* beta_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    NonParaReLUSigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, beta_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(NonParaReLUSigmoidLayer);


}  // namespace caffe
