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
#include <pthread.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#define resize_width 280
#define resize_height 210
#define random(x) (rand()%x)
#define para_dim 14     // dimension of output
// #define total_frames 484815 //484815
#define total_frames 10000 // training, validation, test size
//////////////// by chenyi

namespace caffe {

template <typename Dtype>
DataLstmTrainHistLayer<Dtype>::~DataLstmTrainHistLayer<Dtype>() {
  this->JoinPrefetchThread();
}


//IplImage* leveldbTrain=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);

template <typename Dtype>
void DataLstmTrainHistLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.max_open_files = 100;
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_lstm_train_hist_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_lstm_train_hist_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << this->layer_param_.data_lstm_train_hist_param().source() << std::endl
                     << status.ToString();
  db_.reset(db_temp);

  // Read a data point, to initialize the prefetch and top blobs.
  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  srand((int)time(0)); 

  snprintf(key_cstr, kMaxKeyLength, "%08d", 1);
  db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
  Datum datum;
  datum.ParseFromString(value);

  const int ind_seq_num=this->layer_param_.data_lstm_train_hist_param().sequence_num();

  buffer_key.clear();
  buffer_marker.clear();
  buffer_total.clear();
  hist_blob.clear();

  for (int i=0;i<ind_seq_num;i++) {
     int tmp=random(total_frames)+1;
     buffer_key.push_back(tmp);
     buffer_marker.push_back(0);
     buffer_total.push_back(0);
     for (int j = 0; j < para_dim; ++j) 
         hist_blob.push_back(0);
  }

  int batch_size=this->layer_param_.data_lstm_train_hist_param().sequence_size()*ind_seq_num;
  // image 
  top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
  this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  top[1]->Reshape(batch_size, 1, 1, para_dim);
  this->prefetch_label_.Reshape(batch_size, 1, 1, para_dim);

  // hist
  top[2]->Reshape(batch_size, 1, 1, para_dim);
  this->prefetch_hist_.Reshape(batch_size, 1, 1, para_dim);

  // marker
  top[3]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_marker_.Reshape(batch_size, 1, 1, 1);

  const string& mean_file = this->layer_param_.data_lstm_train_hist_param().mean_file();
  LOG(INFO) << "Loading mean file from: " << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  data_mean_.FromProto(blob_proto);
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLstmTrainHistLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());

  Datum datum;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_hist = this->prefetch_hist_.mutable_cpu_data();
  Dtype* top_marker = this->prefetch_marker_.mutable_cpu_data();

  // datum scales
  const int size = resize_height*resize_width*3;
  const Dtype* mean = this->data_mean_.mutable_cpu_data();

  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int key;

  const int sequence_size = this->layer_param_.data_lstm_train_hist_param().sequence_size();
  const int ind_seq_num=this->layer_param_.data_lstm_train_hist_param().sequence_num();
  const int interval=this->layer_param_.data_lstm_train_hist_param().interval();
  int item_id;

  for (int time_id = 0; time_id < sequence_size; ++time_id) {
     for (int seq_id = 0; seq_id < ind_seq_num; ++seq_id) {
        item_id=time_id*ind_seq_num+seq_id;
        timer.Start();
        // get a blob

        key=buffer_key[seq_id];  // MUST be changed according to the size of the training set

        snprintf(key_cstr, kMaxKeyLength, "%08d", key);
        db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
        datum.ParseFromString(value);
        const string& data = datum.data();

        read_time += timer.MicroSeconds();
        timer.Start();

        for (int j = 0; j < size; ++j) {
           Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[j]));
           top_data[item_id * size + j] = (datum_element - mean[j]);
        }

        for (int j = 0; j < para_dim; ++j) { 
           top_label[item_id * para_dim + j] = datum.float_data(j); 
        }

        top_marker[item_id] = datum.float_data(para_dim);

        if (buffer_marker[seq_id] == 0) {
            top_marker[item_id] = 0;   
            buffer_marker[seq_id] = 1;
        }

        //////////////////////////////////// for hist
        if (top_marker[item_id] < 0.5) {
           for (int j = 0; j < para_dim; ++j)
               top_hist[item_id * para_dim + j] = 0; 
        } else {
           if (time_id == 0) {
              top_hist[item_id * para_dim + 0] = hist_blob[seq_id * para_dim + 0]/1.1+0.5;
              top_hist[item_id * para_dim + 1] = hist_blob[seq_id * para_dim + 1]*0.17778+1.34445;
              top_hist[item_id * para_dim + 2] = hist_blob[seq_id * para_dim + 2]*0.14545+0.39091;
              top_hist[item_id * para_dim + 3] = hist_blob[seq_id * para_dim + 3]*0.17778-0.34445;
              top_hist[item_id * para_dim + 4] = hist_blob[seq_id * para_dim + 4]/95.0+0.12;
              top_hist[item_id * para_dim + 5] = hist_blob[seq_id * para_dim + 5]/95.0+0.12;
              top_hist[item_id * para_dim + 6] = hist_blob[seq_id * para_dim + 6]*0.14545+1.48181;
              top_hist[item_id * para_dim + 7] = hist_blob[seq_id * para_dim + 7]*0.16+0.98;
              top_hist[item_id * para_dim + 8] = hist_blob[seq_id * para_dim + 8]*0.16+0.02;
              top_hist[item_id * para_dim + 9] = hist_blob[seq_id * para_dim + 9]*0.14545-0.48181;
              top_hist[item_id * para_dim + 10] = hist_blob[seq_id * para_dim + 10]/95.0+0.12;
              top_hist[item_id * para_dim + 11] = hist_blob[seq_id * para_dim + 11]/95.0+0.12;
              top_hist[item_id * para_dim + 12] = hist_blob[seq_id * para_dim + 12]/95.0+0.12;
              top_hist[item_id * para_dim + 13] = hist_blob[seq_id * para_dim + 13]*0.6+0.2;
           } else {
              int pre_id=(time_id-1)*ind_seq_num+seq_id;
              top_hist[item_id * para_dim + 0] = top_label[pre_id * para_dim + 0]/1.1+0.5;
              top_hist[item_id * para_dim + 1] = top_label[pre_id * para_dim + 1]*0.17778+1.34445;
              top_hist[item_id * para_dim + 2] = top_label[pre_id * para_dim + 2]*0.14545+0.39091;
              top_hist[item_id * para_dim + 3] = top_label[pre_id * para_dim + 3]*0.17778-0.34445;
              top_hist[item_id * para_dim + 4] = top_label[pre_id * para_dim + 4]/95.0+0.12;
              top_hist[item_id * para_dim + 5] = top_label[pre_id * para_dim + 5]/95.0+0.12;
              top_hist[item_id * para_dim + 6] = top_label[pre_id * para_dim + 6]*0.14545+1.48181;
              top_hist[item_id * para_dim + 7] = top_label[pre_id * para_dim + 7]*0.16+0.98;
              top_hist[item_id * para_dim + 8] = top_label[pre_id * para_dim + 8]*0.16+0.02;
              top_hist[item_id * para_dim + 9] = top_label[pre_id * para_dim + 9]*0.14545-0.48181;
              top_hist[item_id * para_dim + 10] = top_label[pre_id * para_dim + 10]/95.0+0.12;
              top_hist[item_id * para_dim + 11] = top_label[pre_id * para_dim + 11]/95.0+0.12;
              top_hist[item_id * para_dim + 12] = top_label[pre_id * para_dim + 12]/95.0+0.12;
              top_hist[item_id * para_dim + 13] = top_label[pre_id * para_dim + 13]*0.6+0.2;
           }
        }
        //////////////////////////////////// for hist

        trans_time += timer.MicroSeconds();

        buffer_key[seq_id]++;
        buffer_total[seq_id]++;
        if (buffer_key[seq_id]>total_frames || buffer_total[seq_id]>interval) {
           buffer_key[seq_id]=random(total_frames)+1;
           buffer_marker[seq_id]=0;
           buffer_total[seq_id]=0;
        }

        //////////////////////////////////// for hist
        if (time_id==sequence_size-1) {
           for (int j = 0; j < para_dim; ++j) 
               hist_blob[seq_id * para_dim + j] = datum.float_data(j); 
        }
        //////////////////////////////////// for hist

/*
        if (seq_id == 0) {
           for (int h = 0; h < resize_height; ++h) {
              for (int w = 0; w < resize_width; ++w) {
                 leveldbTrain->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
                 leveldbTrain->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
                 leveldbTrain->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];

                 //leveldbTrain->imageData[(h*resize_width+w)*3+0]=(uint8_t)top_data[item_id * size+h*resize_width+w];
                 //leveldbTrain->imageData[(h*resize_width+w)*3+1]=(uint8_t)top_data[item_id * size+resize_height*resize_width+h*resize_width+w];
                 //leveldbTrain->imageData[(h*resize_width+w)*3+2]=(uint8_t)top_data[item_id * size+resize_height*resize_width*2+h*resize_width+w];
               }
           }
           cvShowImage("Image from leveldb", leveldbTrain);
           cvWaitKey( 1 );
        }
*/
     }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLstmTrainHistLayer);
REGISTER_LAYER_CLASS(DataLstmTrainHist);

}  // namespace caffe
