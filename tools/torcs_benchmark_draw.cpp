/////////////////////////////////////////////////
///
/// draw the results of different CNN models on the same testing set, for frame by frame visual comparison.  Only for 2-lane setting.
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

#define resize_width 640
#define resize_height 480
#define semantic_width 320
#define semantic_height 660

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    ////////////////////// set up opencv
    IplImage* leveldbRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    IplImage* semanticRGB=cvCreateImage(cvSize(semantic_width,semantic_height),IPL_DEPTH_8U,3);
    IplImage* legend=cvLoadImage("../torcs/Legend1.png");
    IplImage* background=cvLoadImage("../torcs/semantic_background_2lane.png");
    cvNamedWindow("Visualization CNN 1F",1);
    cvNamedWindow("Visualization LSTM 1",1);
    cvNamedWindow("Visualization LSTM 2",1);
    cvNamedWindow("Image from leveldb",1);
    cvNamedWindow("Legend",1);

    cvShowImage("Legend",legend);

    int key;

    CvFont font;    
    cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 1, 1, 1, 2, 8);  
    char vi_buf[20];
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

    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    ////////////////////// set up leveldb

    int frame = 0;

    float true_angle;
    float true_toMarking_L;
    float true_toMarking_M;
    float true_toMarking_R;
    float true_dist_L;
    float true_dist_R;
    float true_toMarking_LL;
    float true_toMarking_ML;
    float true_toMarking_MR;
    float true_toMarking_RR;
    float true_dist_LL;
    float true_dist_MM;
    float true_dist_RR;

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
    ////////////////////// visualization parameters
    int marking_head=1;
    int marking_st;
    int marking_end;
    int pace;
    int car_pos;

    float p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y;
    CvPoint* pt = new CvPoint[4]; 
    int visualize_angle=1;
    ////////////////////// visualization parameters

    FILE *fp1, *fp2, *fp3;
    fp1=fopen("result_pre_trained/cnn_gist.txt","rb");
    fp2=fopen("result_lstm_11/cnn_gist.txt","rb");
    fp3=fopen("result_lstm_12/cnn_gist.txt","rb");
    char sbuf[50];
    float digit; 

    while (frame<8639) {  
       // 2lane_aalborg_apline2: 37602, 2lane_dirt3: 16454, 2lane_etrack2: 28419, 2lane_wheel2: 32185, result_48.1: 86564
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

       true_angle=-true_angle;
     
       for (int h = 0; h < resize_height; ++h) {
           for (int w = 0; w < resize_width; ++w) {
               leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
               leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
           }
       }
       cvShowImage("Image from leveldb", leveldbRGB);


       for (int i=1;i<=3;i++) {

            if (i==1) {

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               angle=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_L=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_M=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_R=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               dist_L=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               dist_R=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_LL=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_ML=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_MR=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_RR=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               dist_LL=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               dist_MM=digit;

               fgets(sbuf,50,fp1);
               sscanf(sbuf, "%f", &digit ); 
               dist_RR=digit;

               printf("CNN 1F: %f,%f,%f,%f,%f,%f,%f\n", dist_LL, dist_MM, dist_RR, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
               printf("CNN 1F: %f,%f,%f,%f,%f,%f\n\n", dist_L, dist_R, toMarking_L, toMarking_M, toMarking_R, angle);
               fflush(stdout);

            } else if (i==2) {

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               angle=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_L=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_M=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_R=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               dist_L=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               dist_R=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_LL=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_ML=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_MR=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_RR=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               dist_LL=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               dist_MM=digit;

               fgets(sbuf,50,fp2);
               sscanf(sbuf, "%f", &digit ); 
               dist_RR=digit;

               printf("LSTM 1: %f,%f,%f,%f,%f,%f,%f\n", dist_LL, dist_MM, dist_RR, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
               printf("LSTM 1: %f,%f,%f,%f,%f,%f\n\n", dist_L, dist_R, toMarking_L, toMarking_M, toMarking_R, angle);
               fflush(stdout);

            } else {

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               angle=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_L=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_M=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_R=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               dist_L=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               dist_R=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_LL=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_ML=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_MR=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               toMarking_RR=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               dist_LL=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               dist_MM=digit;

               fgets(sbuf,50,fp3);
               sscanf(sbuf, "%f", &digit ); 
               dist_RR=digit;

               printf("LSTM 2: %f,%f,%f,%f,%f,%f,%f\n", dist_LL, dist_MM, dist_RR, toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR);
               printf("LSTM 2: %f,%f,%f,%f,%f,%f\n\n", dist_L, dist_R, toMarking_L, toMarking_M, toMarking_R, angle);
               fflush(stdout);
            }

            ///////////////////////////// semantic visualization
            cvCopy(background,semanticRGB);

            pace=20;

            marking_head=marking_head+pace;
            if (marking_head>0) marking_head=marking_head-110;
            else if (marking_head<-110) marking_head=marking_head+110;

            marking_st=marking_head;
            marking_end=marking_head+55;

            while (marking_st<=660) {
                cvLine(semanticRGB,cvPoint(150,marking_st),cvPoint(150,marking_end),cvScalar(255,255,255),2);
                marking_st=marking_st+110;
                marking_end=marking_end+110;
            }

            cvRectangle(semanticRGB,cvPoint(240,90),cvPoint(310,125),cvScalar(2,100,41),-1);

            //////////////// visualize true_angle
            if (visualize_angle==1) {
               p1_x=-14*cos(true_angle)+28*sin(true_angle);
               p1_y=14*sin(true_angle)+28*cos(true_angle);
               p2_x=14*cos(true_angle)+28*sin(true_angle);
               p2_y=-14*sin(true_angle)+28*cos(true_angle);
               p3_x=14*cos(true_angle)-28*sin(true_angle);
               p3_y=-14*sin(true_angle)-28*cos(true_angle);
               p4_x=-14*cos(true_angle)-28*sin(true_angle);
               p4_y=14*sin(true_angle)-28*cos(true_angle);
            }
            //////////////// visualize true_angle

            /////////////////// draw groundtruth data
            if (true_toMarking_LL>-9) {     // right lane
  
               if (true_toMarking_M<2 && true_toMarking_R>6.5)
                   car_pos=int((174-(true_toMarking_ML+true_toMarking_MR)*6+198-true_toMarking_M*12)/2);
               else if (true_toMarking_M<2 && true_toMarking_R<6.5)
                   car_pos=int((174-(true_toMarking_ML+true_toMarking_MR)*6+150-true_toMarking_M*12)/2);
               else
                   car_pos=int(174-(true_toMarking_ML+true_toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                  cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);

               if (true_dist_LL<60)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_LL*12-28),cvPoint(126+14,600-true_dist_LL*12+28),cvScalar(0,255,255),-1);
               if (true_dist_MM<60)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_MM*12-28),cvPoint(174+14,600-true_dist_MM*12+28),cvScalar(0,255,255),-1);
            }

            else if (true_toMarking_RR<9) {   // left lane

               if (true_toMarking_M<2 && true_toMarking_L<-6.5)
                   car_pos=int((126-(true_toMarking_ML+true_toMarking_MR)*6+102-true_toMarking_M*12)/2);
               else if (true_toMarking_M<2 && true_toMarking_L>-6.5)
                   car_pos=int((126-(true_toMarking_ML+true_toMarking_MR)*6+150-true_toMarking_M*12)/2);
               else
                   car_pos=int(126-(true_toMarking_ML+true_toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                  cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);

               if (true_dist_MM<60)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_MM*12-28),cvPoint(126+14,600-true_dist_MM*12+28),cvScalar(0,255,255),-1);
               if (true_dist_RR<60)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_RR*12-28),cvPoint(174+14,600-true_dist_RR*12+28),cvScalar(0,255,255),-1);
            }

            else if (true_toMarking_M<3) {
                if (true_toMarking_L<-6.5) {   // left
                   car_pos=int(102-true_toMarking_M*12);
                   if (true_dist_R<60)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_R*12-28),cvPoint(126+14,600-true_dist_R*12+28),cvScalar(0,255,255),-1);
                } else if (true_toMarking_R>6.5) {  // right
                   car_pos=int(198-true_toMarking_M*12);
                   if (true_dist_L<60)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_L*12-28),cvPoint(174+14,600-true_dist_L*12+28),cvScalar(0,255,255),-1);
                } else {
                   car_pos=int(150-true_toMarking_M*12);
                   if (true_dist_L<60)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_L*12-28),cvPoint(126+14,600-true_dist_L*12+28),cvScalar(0,255,255),-1);
                   if (true_dist_R<60)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_R*12-28),cvPoint(174+14,600-true_dist_R*12+28),cvScalar(0,255,255),-1);
                }

                if (visualize_angle==1) {
                   pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                   pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                   pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                   pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                   cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
                } else
                   cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);
            }
            /////////////////// draw groundtruth data

            //////////////// visualize angle
            if (visualize_angle==1) {
               angle=-angle;
               p1_x=-14*cos(angle)+28*sin(angle);
               p1_y=14*sin(angle)+28*cos(angle);
               p2_x=14*cos(angle)+28*sin(angle);
               p2_y=-14*sin(angle)+28*cos(angle);
               p3_x=14*cos(angle)-28*sin(angle);
               p3_y=-14*sin(angle)-28*cos(angle);
               p4_x=-14*cos(angle)-28*sin(angle);
               p4_y=14*sin(angle)-28*cos(angle);
            }
            //////////////// visualize angle

            /////////////////// draw sensing data
            if (toMarking_LL>-8 && toMarking_RR>8 && -toMarking_ML+toMarking_MR<5.5) {     // right lane
 
               if (toMarking_M<1.5 && toMarking_R>6)
                   car_pos=int((174-(toMarking_ML+toMarking_MR)*6+198-toMarking_M*12)/2);
               else if (toMarking_M<1.5 && toMarking_R<=6)
                   car_pos=int((174-(toMarking_ML+toMarking_MR)*6+150-toMarking_M*12)/2);
               else
                   car_pos=int(174-(toMarking_ML+toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                  int npts=4;
                  cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);

               if (dist_LL<50)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-dist_LL*12-28),cvPoint(126+14,600-dist_LL*12+28),cvScalar(237,99,157),2);
               if (dist_MM<50)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-dist_MM*12-28),cvPoint(174+14,600-dist_MM*12+28),cvScalar(237,99,157),2);
            }

            else if (toMarking_RR<8 && toMarking_LL<-8 && -toMarking_ML+toMarking_MR<5.5) {   // left lane

               if (toMarking_M<1.5 && toMarking_L<-6)
                   car_pos=int((126-(toMarking_ML+toMarking_MR)*6+102-toMarking_M*12)/2);
               else if (toMarking_M<1.5 && toMarking_L>=-6)
                   car_pos=int((126-(toMarking_ML+toMarking_MR)*6+150-toMarking_M*12)/2);
               else
                   car_pos=int(126-(toMarking_ML+toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                  int npts=4;
                  cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);

               if (dist_MM<50)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-dist_MM*12-28),cvPoint(126+14,600-dist_MM*12+28),cvScalar(237,99,157),2);
               if (dist_RR<50)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-dist_RR*12-28),cvPoint(174+14,600-dist_RR*12+28),cvScalar(237,99,157),2);
            }

            else if (toMarking_M<2.5) {
                if (toMarking_L<-6) {   // left
                   car_pos=int(102-toMarking_M*12);
                   if (dist_R<50)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-dist_R*12-28),cvPoint(126+14,600-dist_R*12+28),cvScalar(237,99,157),2);
                } else if (toMarking_R>6) {  // right
                   car_pos=int(198-toMarking_M*12);
                   if (dist_L<50)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-dist_L*12-28),cvPoint(174+14,600-dist_L*12+28),cvScalar(237,99,157),2);
                } else if (toMarking_R<6 && toMarking_L>-6) {
                   car_pos=int(150-toMarking_M*12);
                   if (dist_L<50)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-dist_L*12-28),cvPoint(126+14,600-dist_L*12+28),cvScalar(237,99,157),2);
                   if (dist_R<50)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-dist_R*12-28),cvPoint(174+14,600-dist_R*12+28),cvScalar(237,99,157),2);
                }

                if (visualize_angle==1) {
                   pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                   pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                   pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                   pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                   int npts=4;
                   cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);
                } else
                   cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);
            }
            /////////////////// draw sensing data

            if (i==1)
                sprintf(vi_buf,"CNN 1F");
            else if (i==2)
                sprintf(vi_buf,"LSTM 1");
            else
                sprintf(vi_buf,"LSTM 2");
            cvPutText(semanticRGB,vi_buf,cvPoint(20,40),&font,cvScalar(0,0,255));
            
            if (i==1)
               cvShowImage("Visualization CNN 1F",semanticRGB);
            else if (i==2)
               cvShowImage("Visualization LSTM 1",semanticRGB);
            else
               cvShowImage("Visualization LSTM 2",semanticRGB);
            ///////////////////////////// semantic visualization

       }

       key=cvWaitKey( 100 );

       //////////////////////// Linux
       if (key==1048603)
          break;  // esc 和 window 下不一样 
       //////////////////////// Linux

       //////////////////////// Windows
       else if (key==27) 
          break;
       //////////////////////// Windows
    }  // end while (frame<3600)

    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvDestroyWindow("Visualization CNN 1F");
    cvDestroyWindow("Visualization LSTM 1");
    cvDestroyWindow("Visualization LSTM 2");
    cvDestroyWindow("Legend");
    cvReleaseImage( &leveldbRGB );
    cvReleaseImage( &semanticRGB );
    cvReleaseImage( &background );
    cvReleaseImage( &legend );
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete db;
    ////////////////////// clean up leveldb
}
