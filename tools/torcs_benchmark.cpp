/////////////////////////////////////////////////
///
/// test the trained CNN on benchmark testing set.
///
/////////////////////////////////////////////////

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <unistd.h>  
#include <stdlib.h>  
#include <stdio.h>  
#include <sys/shm.h>
#include <cuda_runtime.h>
#include <cstring>
#include <math.h>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define image_width 640
#define image_height 480
#define resize_width 280
#define resize_height 210

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    ////////////////////// set up opencv
    IplImage* imageRGB=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
    IplImage* resizeRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    cvNamedWindow("Original Image",1);
    cvNamedWindow("Image from leveldb",1);
    int key;
    ////////////////////// set up opencv

    ////////////////////// set up leveldb  
    leveldb::Options options;
    options.error_if_exists = false;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;

    leveldb::DB* db;
    LOG(INFO) << "Opening leveldb: TORCS_Testing_1F";
    leveldb::Status status = leveldb::DB::Open(options, "/D/TORCS_Trainset/TORCS_GIST_1F_Testing", &db);
    CHECK(status.ok()) << "Failed to open leveldb: TORCS_Testing_1F";
    Datum datum;

    leveldb::DB* db2;
    LOG(INFO) << "Opening leveldb: Current_State_1F";
    leveldb::Status status2 = leveldb::DB::Open(options, "Current_State_1F", &db2);
    CHECK(status2.ok()) << "Failed to open leveldb: Current_State_1F";
    Datum datum2;

    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();
    ////////////////////// set up leveldb

    ////////////////////// set up Caffe
    if (argc < 3) {
      LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations " << "[CPU/GPU]";
      return 0;
    }

    cudaSetDevice(0);

    if (argc == 4 && strcmp(argv[3], "GPU") == 0) {
      LOG(ERROR) << "Using GPU";
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(ERROR) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }

    NetParameter test_net_param;
    ReadProtoFromTextFile(argv[1], &test_net_param);
    Net<float> caffe_test_net(test_net_param, db2);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(argv[2], &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    vector<Blob<float>*> dummy_blob_input_vec;
    ////////////////////// set up Caffe

    ////////////////////// cnn output parameters
    float true_angle;
    int true_fast;

    float true_dist_L;
    float true_dist_R;

    float true_toMarking_L;
    float true_toMarking_M;
    float true_toMarking_R;

    float true_dist_LL;
    float true_dist_MM;
    float true_dist_RR;

    float true_toMarking_LL;
    float true_toMarking_ML;
    float true_toMarking_MR;
    float true_toMarking_RR;
    float marker;

    //////

    float angle;
    int fast;

    float dist_L;
    float dist_R;

    float toMarking_L;
    float toMarking_M;
    float toMarking_R;

    float dist_LL;
    float dist_MM;
    float dist_RR;

    float toMarking_LL;
    float toMarking_ML;
    float toMarking_MR;
    float toMarking_RR;
    ////////////////////// cnn output parameters

    int frame = 0;

    FILE *fp1, *fp2, *fp3;
    fp1=fopen("cnn_gist.txt","wb");
    fp2=fopen("err_cnn_gist.txt","wb");
    //fp3=fopen("err_gt_gist.txt","wb");
    char sbuf[1000]; 

    while (frame<8639) {  
    //TORCS_Testing_1F_Baseline: 8639, TORCS_Caltech_1F_Training: 2430, TORCS_Caltech_1F_Testing: 2533

       frame++; 
      
       ///////////////////////////// read leveldb
       snprintf(key_cstr, kMaxKeyLength, "%08d", frame);
       db->Get(leveldb::ReadOptions(), string(key_cstr), &value);
       datum.ParseFromString(value);
       const string& data = datum.data();

       true_angle=datum.float_data(0);
       true_toMarking_L=datum.float_data(1);
       true_toMarking_M=datum.float_data(2);
       true_toMarking_R=datum.float_data(3);
       true_dist_L=datum.float_data(4);
       true_dist_R=datum.float_data(5);
       true_toMarking_LL=datum.float_data(6);
       true_toMarking_ML=datum.float_data(7);
       true_toMarking_MR=datum.float_data(8);
       true_toMarking_RR=datum.float_data(9);
       true_dist_LL=datum.float_data(10);
       true_dist_MM=datum.float_data(11);
       true_dist_RR=datum.float_data(12);
       true_fast=datum.float_data(13);
       marker=datum.float_data(14);
    
       for (int h = 0; h < image_height; ++h) {
           for (int w = 0; w < image_width; ++w) {
               imageRGB->imageData[(h*image_width+w)*3+0]=(uint8_t)data[h*image_width+w];
               imageRGB->imageData[(h*image_width+w)*3+1]=(uint8_t)data[image_height*image_width+h*image_width+w];
               imageRGB->imageData[(h*image_width+w)*3+2]=(uint8_t)data[image_height*image_width*2+h*image_width+w];
           }
       }
       cvShowImage("Original Image", imageRGB);
       cvResize(imageRGB,resizeRGB);

            ///////////////////////////// set caffe input
            datum2.set_channels(3);
            datum2.set_height(resize_height);
            datum2.set_width(resize_width);
            datum2.set_label(0); 
            datum2.clear_data();
            datum2.clear_float_data();
            string* datum2_string = datum2.mutable_data();

            for (int c = 0; c < 3; ++c) {
              for (int h = 0; h < resize_height; ++h) {
                for (int w = 0; w < resize_width; ++w) {
                  datum2_string->push_back(static_cast<char>(resizeRGB->imageData[(h*resize_width+w)*3+c]));
                }
              }
            }

            datum2.add_float_data(marker); 

            datum2.SerializeToString(&value);

            for (int i=1; i<=1; i++) {
               snprintf(key_cstr, kMaxKeyLength, "%08d", i);
               batch->Put(string(key_cstr), value);
            }
            db2->Write(leveldb::WriteOptions(), batch);
            delete batch;
            batch = new leveldb::WriteBatch();
            ///////////////////////////// set caffe input

            ///////////////////////////// determine the next action
            const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);  

            const float* result_data = result[0]->cpu_data();

            angle=(result_data[0]-0.5)*1.1;

            toMarking_L=(result_data[1]-1.34445)*5.6249;
            toMarking_M=(result_data[2]-0.39091)*6.8752;
            toMarking_R=(result_data[3]+0.34445)*5.6249;

            dist_L=(result_data[4]-0.12)*95;
            dist_R=(result_data[5]-0.12)*95;

            toMarking_LL=(result_data[6]-1.48181)*6.8752;
            toMarking_ML=(result_data[7]-0.98)*6.25;
            toMarking_MR=(result_data[8]-0.02)*6.25;
            toMarking_RR=(result_data[9]+0.48181)*6.8752;

            dist_LL=(result_data[10]-0.12)*95;
            dist_MM=(result_data[11]-0.12)*95;
            dist_RR=(result_data[12]-0.12)*95;

            if (result_data[13]>0.5) fast=1;
            else fast=0;

            printf("M_LL:%.2lf, M_ML:%.2lf, M_MR:%.2lf, M_RR:%.2lf\n", toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
            printf("M_L:%.2lf, M_M:%.2lf, M_R:%.2lf, angle:%.3lf\n", toMarking_L, toMarking_M, toMarking_R, angle);
            fflush(stdout);

            sprintf(sbuf, "%.4lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n%.3lf\n", angle, toMarking_L, toMarking_M, toMarking_R, dist_L, dist_R, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR, dist_LL, dist_MM, dist_RR);
            fputs(sbuf,fp1);

            sprintf(sbuf, "%.4lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf\n", angle, toMarking_L, toMarking_M, toMarking_R, dist_L, dist_R, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR, dist_LL, dist_MM, dist_RR);
            fputs(sbuf,fp2);
/*
            sprintf(sbuf, "%.4lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf\n", true_angle, true_toMarking_L, true_toMarking_M, true_toMarking_R, true_dist_L, true_dist_R, true_toMarking_LL, true_toMarking_ML, true_toMarking_MR, true_toMarking_RR, true_dist_LL, true_dist_MM, true_dist_RR);
            fputs(sbuf,fp3);
*/
            ///////////////////////////// determine the next action

       key=cvWaitKey( 1 );

       //////////////////////// Linux
       if (key==1048603)
          break;  // esc 和 window 下不一样 
       //////////////////////// Linux

    }  // end while (frame<3600)

    fclose(fp1);
    fclose(fp2);
    //fclose(fp3);

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvDestroyWindow("Original Image");
    cvReleaseImage( &imageRGB );
    cvReleaseImage( &resizeRGB );
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete batch;
    //delete db2;
    delete db;
    ////////////////////// clean up leveldb
}
