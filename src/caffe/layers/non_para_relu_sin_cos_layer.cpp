#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NonParaReLUSinCosLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  NonParaReLUSinCosParameter nprelu_param = this->layer_param().nprelu_sin_cos_param();
  int channels = bottom[0]->channels();
  channel_shared_ = nprelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (nprelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(nprelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.0);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
    filler->Fill(this->blobs_[1].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_sin_.Reshape(vector<int>(1, bottom[0]->count() / bottom[0]->num()));
  caffe_set(multiplier_sin_.count(), Dtype(1), multiplier_sin_.mutable_cpu_data());
  multiplier_cos_.Reshape(vector<int>(1, bottom[0]->count() / bottom[0]->num()));
  caffe_set(multiplier_cos_.count(), Dtype(1), multiplier_cos_.mutable_cpu_data());
}

template <typename Dtype>
void NonParaReLUSinCosLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void NonParaReLUSinCosLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* beta_sin_data = this->blobs_[0]->cpu_data();
  const Dtype* beta_cos_data = this->blobs_[1]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + beta_sin_data[c] * sin(bottom_data[i])
        + beta_cos_data[c] * cos(bottom_data[i]);
  }
}

template <typename Dtype>
void NonParaReLUSinCosLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* beta_sin_data = this->blobs_[0]->cpu_data();
  const Dtype* beta_cos_data = this->blobs_[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0] && this->param_propagate_down_[1]) {
    Dtype* beta_sin_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* beta_cos_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), beta_sin_diff);
    caffe_set(this->blobs_[1]->count(), Dtype(0), beta_cos_diff);
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      beta_sin_diff[c] += top_diff[i] * sin(bottom_data[i]);
      beta_cos_diff[c] += top_diff[i] * cos(bottom_data[i]);
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] * (
          ( bottom_data[i] > 0)
          + beta_sin_data[c] * cos(bottom_data[i])
          - beta_cos_data[c] * sin(bottom_data[i]));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NonParaReLUSinCosLayer);
#endif

INSTANTIATE_CLASS(NonParaReLUSinCosLayer);
REGISTER_LAYER_CLASS(NonParaReLUSinCos);

}  // namespace caffe
