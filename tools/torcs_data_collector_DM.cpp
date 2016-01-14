/////////////////////////////////////////////////
///
/// same frames and driving controls (behavior reflex) while the AI is driving.
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

struct shared_use_st  
{  
    int written;//作为一个标志，非0：表示可读，0表示可写
    uint8_t data[image_width*image_height*3];//记录写入和读取的文本  
    int control;
    int pause;
    double fast;

    double dist_L;
    double dist_R;

    double toMarking_L;
    double toMarking_M;
    double toMarking_R;

    double dist_LL;
    double dist_MM;
    double dist_RR;

    double toMarking_LL;
    double toMarking_ML;
    double toMarking_MR;
    double toMarking_RR;

    double toMiddle;
    double angle;
    double speed;

    double steerCmd;
    double accelCmd;
    double brakeCmd;
};

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    ////////////////////// set up memory sharing
    void *shm = NULL;//分配的共享内存的原始首地址  
    struct shared_use_st *shared;//指向shm  
    int shmid;//共享内存标识符  
    //创建共享内存  
    shmid = shmget((key_t)4567, sizeof(struct shared_use_st), 0666|IPC_CREAT);  
    if(shmid == -1)  
    {  
        fprintf(stderr, "shmget failed\n");  
        exit(EXIT_FAILURE);  
    }  
    //将共享内存连接到当前进程的地址空间  
    shm = shmat(shmid, 0, 0);  
    if(shm == (void*)-1)  
    {  
        fprintf(stderr, "shmat failed\n");  
        exit(EXIT_FAILURE);  
    }  
    printf("\n********** Memory sharing started, attached at %X **********\n", shm); 
    //设置共享内存  
    shared = (struct shared_use_st*)shm;  
    shared->written = 0;
    shared->control = 0;
    shared->pause = 0;
    shared->fast = 0.0;

    shared->dist_L = 0.0;
    shared->dist_R = 0.0;

    shared->toMarking_L = 0.0;
    shared->toMarking_M = 0.0;
    shared->toMarking_R = 0.0;

    shared->dist_LL = 0.0;
    shared->dist_MM = 0.0;
    shared->dist_RR = 0.0;

    shared->toMarking_LL = 0.0;
    shared->toMarking_ML = 0.0;
    shared->toMarking_MR = 0.0;
    shared->toMarking_RR = 0.0;

    shared->toMiddle = 0.0;
    shared->angle = 0.0;
    shared->speed = 0.0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;  
    ////////////////////// set up memory sharing

    ////////////////////// set up opencv
    IplImage* screenRGB=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
    IplImage* resizeRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    IplImage* leveldbRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    cvNamedWindow("Image from leveldb",1);
    cvNamedWindow("Image from TORCS",1);
    int key;
    ////////////////////// set up opencv

    ////////////////////// set up leveldb  
    leveldb::Options options;
    options.error_if_exists = false;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;

    leveldb::DB* db;
    LOG(INFO) << "Opening leveldb: TORCS_DM_1F";
    leveldb::Status status = leveldb::DB::Open(options, "/D/TORCS_DM_1F", &db);
    CHECK(status.ok()) << "Failed to open leveldb: TORCS_DM_1F";

    Datum datum;
    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();
    ////////////////////// set up leveldb

    ////////////////////// cnn output parameters
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

    ////////////////////// control parameters
    float road_width=8.0;
    float center_line;
    float coe_steer=1.0;
    int lane_change=0;
    float pre_ML;
    float pre_MR;
    float desired_speed;
    float steering_record[5]={0,0,0,0,0};
    int steering_head=0;
    float slow_down=100;
    float dist_LL_record=30;
    float dist_RR_record=30;

    int left_clear=0;
    int right_clear=0;
    int left_timer=0;
    int right_timer=0;
    int timer_set=60;

    float pre_dist_L=60;
    float pre_dist_R=60;
    float steer_trend;
    ////////////////////// control parameters

    int frame = 0;
    int frame_offset = 0;
    int new_start = 0;
    float throttle = 0;

    while (1) {

       if (shared->written == 1) {

         frame++;

         for (int h = 0; h < image_height; h++) {
            for (int w = 0; w < image_width; w++) {
               screenRGB->imageData[(h*image_width+w)*3+2]=shared->data[((image_height-h-1)*image_width+w)*3+0];
               screenRGB->imageData[(h*image_width+w)*3+1]=shared->data[((image_height-h-1)*image_width+w)*3+1];
               screenRGB->imageData[(h*image_width+w)*3+0]=shared->data[((image_height-h-1)*image_width+w)*3+2];
            }
         }
        
         cvResize(screenRGB,resizeRGB);
         cvShowImage("Image from TORCS", screenRGB);


               //////////////////////////// controller drive
               angle = shared->angle;           
               fast = int(shared->fast);
               dist_L = shared->dist_L;
               dist_R = shared->dist_R;

               toMarking_L = shared->toMarking_L;
               toMarking_M = shared->toMarking_M;
               toMarking_R = shared->toMarking_R;

               dist_LL = shared->dist_LL;
               dist_MM = shared->dist_MM;
               dist_RR = shared->dist_RR;

               toMarking_LL = shared->toMarking_LL;
               toMarking_ML = shared->toMarking_ML;
               toMarking_MR = shared->toMarking_MR;
               toMarking_RR = shared->toMarking_RR;

               slow_down=100; 

               if (pre_dist_L<20 && dist_LL<20) {
                   left_clear=0;
                   left_timer=0;
               } else left_timer++;                
        
               if (pre_dist_R<20 && dist_RR<20) {
                   right_clear=0;
                   right_timer=0;
               } else right_timer++;

               pre_dist_L=dist_LL;
               pre_dist_R=dist_RR;

               if (left_timer>timer_set) {
                  left_timer=timer_set;
                  left_clear=1;
               }

               if (right_timer>timer_set) {
                  right_timer=timer_set;
                  right_clear=1;
               }
      

               if (lane_change==0 && dist_MM<15) {

                  steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];

                  if (toMarking_LL>-8 && left_clear==1 && steer_trend>=0) {
                     lane_change=-2;
                     coe_steer=6;
                     right_clear=0;
                     right_timer=0;
                     left_clear=0;
                     left_timer=0;
                     timer_set=30;
                  }

                  else if (toMarking_RR<8 && right_clear==1 && steer_trend<=0) {
                     lane_change=2;
                     coe_steer=6;
                     left_clear=0;
                     left_timer=0;
                     right_clear=0;
                     right_timer=0;
                     timer_set=30;
                  }

                  else {
                     float v_max=20;
                     float c=2.772;
                     float d=-0.693;
                     slow_down=v_max*(1-exp(-c/v_max*dist_MM-d));  // optimal vilcity car-following model
                     if (slow_down<0) slow_down=0;
                  }
               }
 
               ///////////////////////////////////////////////// prefer to stay in the right lane
               else if (lane_change==0 && dist_MM>=15) {

                  steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];  // am I turning or not

                  if (toMarking_LL<-8 && right_clear==1 && steer_trend<=0 && steer_trend>-0.2) {  // in left lane, move to right lane
                     lane_change=2;
                     coe_steer=6;
                     right_clear=0;
                     right_timer=20;
                  }
               }
               ///////////////////////////////////////////////// prefer to stay in the right lane

               if (lane_change==0) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     coe_steer=1.5;
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     pre_ML=toMarking_ML;
                     pre_MR=toMarking_MR;
                     if (toMarking_M<1)
                        coe_steer=0.4;
                  } else {
                     if (-pre_ML>pre_MR)
                        center_line=(toMarking_L+toMarking_M)/2;
                     else
                        center_line=(toMarking_R+toMarking_M)/2;
                     coe_steer=0.3;
                  }
               }

               else if (lane_change==-2) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     center_line=(toMarking_LL+toMarking_ML)/2;
                     if (toMarking_L>-5 && toMarking_M<1.5)
                        center_line=(center_line+(toMarking_L+toMarking_M)/2)/2;
                  } else {
                     center_line=(toMarking_L+toMarking_M)/2;
                     coe_steer=10;
                     lane_change=-1;
                  }
               }

               else if (lane_change==-1) {
                  if (toMarking_L>-5 && toMarking_M<1.5) {
                     center_line=(toMarking_L+toMarking_M)/2;
                     if (-toMarking_ML+toMarking_MR<5.5)
                        center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
                  } else {
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     lane_change=0;
                  }
               }

               else if (lane_change==2) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     center_line=(toMarking_RR+toMarking_MR)/2;
                     if (toMarking_R<5 && toMarking_M<1.5)
                        center_line=(center_line+(toMarking_R+toMarking_M)/2)/2;
                  } else {
                     center_line=(toMarking_R+toMarking_M)/2;
                     coe_steer=10;
                     lane_change=1;
                  }
               }

               else if (lane_change==1) {
                  if (toMarking_R<5 && toMarking_M<1.5) {
                     center_line=(toMarking_R+toMarking_M)/2;
                     if (-toMarking_ML+toMarking_MR<5.5)
                        center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
                  } else {
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     lane_change=0;
                  }
               }

               shared->steerCmd = (angle - center_line/road_width) / 0.541052/coe_steer;
 
               if (lane_change==0 && coe_steer>1 && shared->steerCmd>0.1)
                  shared->steerCmd=shared->steerCmd*(2.5*shared->steerCmd+0.75);

               steering_record[steering_head]=shared->steerCmd;
               steering_head++;
               if (steering_head==5) steering_head=0;


               if (fast==1) desired_speed=16;
               else desired_speed=16-fabs(steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4])*4.5;
               if (desired_speed<10) desired_speed=10;

               if (slow_down<desired_speed) desired_speed=slow_down;

               ///////////////////////////// speed control           
               if (desired_speed>=shared->speed) {
                   shared->accelCmd = 0.2*(desired_speed-shared->speed+1);
                   if (shared->accelCmd>1) shared->accelCmd=1.0;
                   shared->brakeCmd = 0.0;
               } else {
                   shared->brakeCmd = 0.1*(shared->speed-desired_speed);
                   if (shared->brakeCmd>1) shared->brakeCmd=1.0;
                   shared->accelCmd = 0.0;
               }
               ///////////////////////////// speed control
               /////////////////////// controller drive


         ///////////////////////////// write primary leveldb     
         datum.set_channels(3);
         datum.set_height(resize_height);
         datum.set_width(resize_width);
         datum.set_label(0); 
         datum.clear_data();
         datum.clear_float_data();
         string* datum_string = datum.mutable_data();

         for (int c = 0; c < 3; ++c) {
           for (int h = 0; h < resize_height; ++h) {
             for (int w = 0; w < resize_width; ++w) {
               datum_string->push_back(static_cast<char>(resizeRGB->imageData[(h*resize_width+w)*3+c]));
             }
           }
         }

         throttle=shared->accelCmd-shared->brakeCmd;
         datum.add_float_data(shared->steerCmd);
         datum.add_float_data(throttle);

         if (new_start == 0) {
            datum.add_float_data(0);
            new_start = 1;
         } else
            datum.add_float_data(1);

         printf("frame: %d\n", frame);
         printf("%f, %f\n\n", shared->steerCmd, throttle);
         fflush(stdout);

         // sequential
         snprintf(key_cstr, kMaxKeyLength, "%08d", frame_offset+frame);
                         
         // get the value
         datum.SerializeToString(&value);
         batch->Put(string(key_cstr), value);
         if (frame % 100 == 0) {
           db->Write(leveldb::WriteOptions(), batch);
           LOG(ERROR) << "Processed " << frame << " files.";
           delete batch;
           batch = new leveldb::WriteBatch();
         }
         ///////////////////////////// write leveldb

         ///////////////////////////// read leveldb
         if (frame>100) {                      
            snprintf(key_cstr, kMaxKeyLength, "%08d", frame_offset+frame-100);
            db->Get(leveldb::ReadOptions(), string(key_cstr), &value);
            datum.ParseFromString(value);
            const string& data = datum.data();
       
            for (int h = 0; h < resize_height; ++h) {
                for (int w = 0; w < resize_width; ++w) {
                    leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
                    leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
                    leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
                }
            }
            cvShowImage("Image from leveldb", leveldbRGB);

         } // end if (frame>100)
         ///////////////////////////// read leveldb 

         shared->written=0;
        }  // if (shared->written == 1)

        key=cvWaitKey( 5 );

        //////////////////////// Linux key
        if (key==1048603 || key==27) {  // Esc: exit
          shared->pause = 0;
          break;  
        }
        else if (key==1048688 || key==112) {  // P: pause
          shared->pause = 1-shared->pause;
          new_start = 0;
        }
        ////// override drive by keyboard
        else if (key==1113938 || key==65362) {
            shared->accelCmd = 1.0;
            shared->brakeCmd = 0;
        }
        else if (key==1113940 || key==65364) {
            shared->brakeCmd = 0.8;
            shared->accelCmd = 0;
        }
        else if (key==1113937 || key==65361) {
            shared->steerCmd = 0.5;
        }
        else if (key==1113939 || key==65363) {
            shared->steerCmd = -0.5;
        }
        //////////////////////// END Linux key

    }  // end while (1) 

    db->Write(leveldb::WriteOptions(), batch);

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from TORCS");
    cvDestroyWindow("Image from leveldb");
    cvReleaseImage( &screenRGB );
    cvReleaseImage( &resizeRGB );
    cvReleaseImage( &leveldbRGB );
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete batch;
    delete db;
    ////////////////////// clean up leveldb

    ////////////////////// clean up memory sharing
    if(shmdt(shm) == -1)  
    {  
        fprintf(stderr, "shmdt failed\n");  
        exit(EXIT_FAILURE);  
    }  
    //删除共享内存  
    if(shmctl(shmid, IPC_RMID, 0) == -1)  
    {  
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        exit(EXIT_FAILURE);  
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    exit(EXIT_SUCCESS); 
    ////////////////////// clean up memory sharing 
}
