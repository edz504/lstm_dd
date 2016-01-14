/// convert original 14D affordance indicator leveldb database to add one float value indicating the clip beginning marker (for lstm)

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

#define resize_width 640
#define resize_height 480

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    ////////////////////// set up opencv
    IplImage* leveldbRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
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
    LOG(INFO) << "Opening leveldb: TORCS_Training_1F";
    leveldb::Status status = leveldb::DB::Open(options, "/D/TORCS_Trainset/TORCS_GIST_1F_Testing", &db);
    CHECK(status.ok()) << "Failed to open leveldb: TORCS_Training_1F";
    Datum datum;

    leveldb::DB* db_smooth;
    LOG(INFO) << "Opening leveldb: TORCS_Training_Smooth_1F";
    leveldb::Status status_smooth = leveldb::DB::Open(options, "/D/TORCS_Trainset/TORCS_GIST_1F_Testing_smooth", &db_smooth);
    CHECK(status_smooth.ok()) << "Failed to open leveldb: TORCS_Training_Smooth_1F";
    Datum datum_smooth;

    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();
    ////////////////////// set up leveldb

    int frame = 0;
    int offset = 0;

    float angle;
    float toMarking_L;
    float toMarking_M;
    float toMarking_R;
    float dist_L;
    float dist_R;
    float toMarking_LL;
    float toMarking_ML;
    float toMarking_MR;
    float toMarking_RR;
    float dist_LL;
    float dist_MM;
    float dist_RR;
    float fast;
    float marker;

    FILE *fp;
    fp=fopen("frames_GIST.txt","rb");  // the last line must be a number larger than the total frame !!!
    char sbuf[50];
    int number;

    fgets(sbuf,50,fp);
    sscanf(sbuf, "%d", &number ); 

    while (frame<8639) {
       frame++; 
      
       if (frame==number) {
          marker=0;
          fgets(sbuf,50,fp);
          sscanf(sbuf, "%d", &number ); 
       } else
          marker=1;

       ///////////////////////////// read leveldb
       snprintf(key_cstr, kMaxKeyLength, "%08d", frame);
       db->Get(leveldb::ReadOptions(), string(key_cstr), &value);
       datum.ParseFromString(value);
       const string& data = datum.data();

       angle=datum.float_data(0);
       toMarking_L=datum.float_data(1);
       toMarking_M=datum.float_data(2);
       toMarking_R=datum.float_data(3);
       dist_L=datum.float_data(4);
       dist_R=datum.float_data(5);
       toMarking_LL=datum.float_data(6);
       toMarking_ML=datum.float_data(7);
       toMarking_MR=datum.float_data(8);
       toMarking_RR=datum.float_data(9);
       dist_LL=datum.float_data(10);
       dist_MM=datum.float_data(11);
       dist_RR=datum.float_data(12);
       fast=datum.float_data(13);
    
       for (int h = 0; h < resize_height; ++h) {
           for (int w = 0; w < resize_width; ++w) {
               leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
           }
       }

       cvShowImage("Image from leveldb", leveldbRGB);

       printf("%d\n", frame);
       printf("%f,%f,%f,%f,%f,%f,%f\n", dist_LL, dist_MM, dist_RR, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
       printf("%f,%f,%f,%f,%f,%f\n\n", dist_L, dist_R, toMarking_L, toMarking_M, toMarking_R, fast);
       fflush(stdout);
       ///////////////////////////// read leveldb

       ///////////////////////////// write leveldb 
       datum_smooth.set_channels(datum.channels());
       datum_smooth.set_height(datum.height());
       datum_smooth.set_width(datum.width());
       datum_smooth.set_label(0);
       datum_smooth.clear_data();
       datum_smooth.clear_float_data();
       string* datum_smooth_string = datum_smooth.mutable_data();
             
       for (int j=0; j<resize_height*resize_width*3; j++)
            datum_smooth_string->push_back(data[j]);

       datum_smooth.add_float_data(angle);
       datum_smooth.add_float_data(toMarking_L);
       datum_smooth.add_float_data(toMarking_M);
       datum_smooth.add_float_data(toMarking_R);
       datum_smooth.add_float_data(dist_L);
       datum_smooth.add_float_data(dist_R);
       datum_smooth.add_float_data(toMarking_LL);
       datum_smooth.add_float_data(toMarking_ML);
       datum_smooth.add_float_data(toMarking_MR);
       datum_smooth.add_float_data(toMarking_RR);
       datum_smooth.add_float_data(dist_LL);
       datum_smooth.add_float_data(dist_MM);
       datum_smooth.add_float_data(dist_RR);
       datum_smooth.add_float_data(fast);
       datum_smooth.add_float_data(marker);  

       snprintf(key_cstr, kMaxKeyLength, "%08d", offset+frame);      
       datum_smooth.SerializeToString(&value);
       batch->Put(string(key_cstr), value);

       if (frame % 100 == 0) {
          db_smooth->Write(leveldb::WriteOptions(), batch);
          delete batch;
          batch = new leveldb::WriteBatch();
       }

       ///////////////////////////// write leveldb 

       key=cvWaitKey( 1 );

       //////////////////////// Linux
       if (key==1048603 || key==27)  // Esc
          break;
       //////////////////////// Linux

    }  // end while (frame<3600)

    db_smooth->Write(leveldb::WriteOptions(), batch);  

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvReleaseImage( &leveldbRGB );
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete batch;
    delete db;
    delete db_smooth;
    ////////////////////// clean up leveldb
}
