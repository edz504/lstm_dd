////////////////////////////////////////////////
//
//  Read the pre-collected leveldb database, display the image and 
//  print the corresponding affordance indicators (show clip beginning marker (lstm), 
//  and can add clip beginning marker)
//
//  Input keys 
//  Esc: exit
//
////////////////////////////////////////////////

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
#define total_frames 8639

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
    leveldb::Status status = leveldb::DB::Open(options, "/D/TORCS_Trainset/TORCS_GIST_1F_Testing", &db);  /// modify the address to leveldb here
    CHECK(status.ok()) << "Failed to open leveldb: TORCS_Training_1F";
    Datum datum;

    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    ////////////////////// set up leveldb

    int frame = 1;

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
/*
    float steering;
    float throttle;
    float marker;
*/
    FILE *fp;
    char sbuf[50]; 
    sprintf(sbuf, "frames.txt");
    fp=fopen(sbuf,"wb");

    while (true) {   ///////////////// total number of images in the database

       printf("frame: %d\n", frame);
       fflush(stdout);

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
       marker=datum.float_data(14);
/*     
       steering=datum.float_data(0);
       throttle=datum.float_data(1);
       marker=datum.float_data(2);
*/
       for (int h = 0; h < resize_height; ++h) {
           for (int w = 0; w < resize_width; ++w) {
               leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
           }
       }
       cvShowImage("Image from leveldb", leveldbRGB);

       printf("%f\n",marker);
       printf("%f,%f,%f,%f,%f,%f,%f\n", dist_LL, dist_MM, dist_RR, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
       printf("%f,%f,%f,%f,%f,%f\n\n", dist_L, dist_R, toMarking_L, toMarking_M, toMarking_R, angle);
       fflush(stdout);
/*
       printf("%f\n",marker);
       printf("%f,%f\n\n", steering, throttle);
       fflush(stdout);
*/
       key=cvWaitKey( 0 );

       //////////////////////// Linux
       if (key==1048603 || key==27)   // exit: Esc
          break;
       else if (key==1113937 || key==65361) {  // left, 1 frame backward
          frame--;
          if (frame<1)
             frame=total_frames;
       }
       else if (key==1113939 || key==65363) {  // right, 1 frame forward
          frame++;
          if (frame>total_frames)
             frame=1;
       }
       else if (key==1113938 || key==65362) {  // up, 2 frames backward
          frame=frame-100;
          if (frame<1)
             frame=total_frames;
       }
       else if (key==1113940 || key==65364) {  // down, 2 frames forward
          frame=frame+100;
          if (frame>total_frames)
             frame=1;
       }

       else if (key==1048675 || key==99) { // C, write down clip beginning marker
          sprintf(sbuf, "%d\n", frame);
          fputs(sbuf,fp);
          printf("FRAME SAVED\n\n");
          fflush(stdout);
       }

       //////////////////////// Linux

    }  // end while

    fclose(fp);

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvReleaseImage( &leveldbRGB );;
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete db;
    ////////////////////// clean up leveldb
}
