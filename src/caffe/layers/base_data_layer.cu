#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

//////////////////////////////////////////////////////////////// by chenyi
template <typename Dtype>
void BasePrefetchingDataLayer2<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());

  top[1]->Reshape(this->prefetch_label_.num(), this->prefetch_label_.channels(),
      this->prefetch_label_.height(), this->prefetch_label_.width());
  caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
      top[1]->mutable_gpu_data());

  top[2]->Reshape(this->prefetch_marker_.num(), this->prefetch_marker_.channels(),
      this->prefetch_marker_.height(), this->prefetch_marker_.width());
  caffe_copy(prefetch_marker_.count(), prefetch_marker_.cpu_data(),
      top[2]->mutable_gpu_data());

  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer3<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());

  top[1]->Reshape(this->prefetch_label_.num(), this->prefetch_label_.channels(),
      this->prefetch_label_.height(), this->prefetch_label_.width());
  caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
      top[1]->mutable_gpu_data());

  top[2]->Reshape(this->prefetch_hist_.num(), this->prefetch_hist_.channels(),
      this->prefetch_hist_.height(), this->prefetch_hist_.width());
  caffe_copy(prefetch_hist_.count(), prefetch_hist_.cpu_data(),
      top[2]->mutable_gpu_data());

  top[3]->Reshape(this->prefetch_marker_.num(), this->prefetch_marker_.channels(),
      this->prefetch_marker_.height(), this->prefetch_marker_.width());
  caffe_copy(prefetch_marker_.count(), prefetch_marker_.cpu_data(),
      top[3]->mutable_gpu_data());

  // Start a new prefetch thread
  CreatePrefetchThread();
}
//////////////////////////////////////////////////////////////// by chenyi

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer2);   ///// by chenyi
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer3);   ///// by chenyi

}  // namespace caffe
