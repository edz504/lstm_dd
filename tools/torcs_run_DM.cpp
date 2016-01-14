/////////////////////////////////////////////////
///
/// autonomous driving in TORCS (behavior reflex).
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
    int written;  //a label, if 1: available to read, if 0: available to write
    uint8_t data[image_width*image_height*3];  // image data field
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
    void *shm = NULL;  
    struct shared_use_st *shared; 
    int shmid; 
  
    shmid = shmget((key_t)4567, sizeof(struct shared_use_st), 0666|IPC_CREAT);  
    if(shmid == -1)  
    {  
        fprintf(stderr, "shmget failed\n");  
        exit(EXIT_FAILURE);  
    }  

    shm = shmat(shmid, 0, 0);  
    if(shm == (void*)-1)  
    {  
        fprintf(stderr, "shmat failed\n");  
        exit(EXIT_FAILURE);  
    }  
    printf("\n********** Memory sharing started, attached at %X **********\n", shm); 

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
    ////////////////////// END set up memory sharing

    ////////////////////// set up opencv
    IplImage* screenRGB=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
    IplImage* resizeRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
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
    LOG(INFO) << "Opening leveldb: Current_State_1F";
    leveldb::Status status = leveldb::DB::Open(options, "Current_State_1F", &db);
    CHECK(status.ok()) << "Failed to open leveldb: Current_State_1F";

    Datum datum;
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
    Net<float> caffe_test_net(test_net_param, db);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(argv[2], &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    vector<Blob<float>*> dummy_blob_input_vec;
    ////////////////////// END set up Caffe

    float desired_speed=60/3.6;

    while (1) {

        if (shared->written == 1) {  // the new image data is ready to be read

            for (int h = 0; h < image_height; h++) {
               for (int w = 0; w < image_width; w++) {
                  screenRGB->imageData[(h*image_width+w)*3+2]=shared->data[((image_height-h-1)*image_width+w)*3+0];
                  screenRGB->imageData[(h*image_width+w)*3+1]=shared->data[((image_height-h-1)*image_width+w)*3+1];
                  screenRGB->imageData[(h*image_width+w)*3+0]=shared->data[((image_height-h-1)*image_width+w)*3+2];
               }
            }

            cvResize(screenRGB,resizeRGB);

            ///////////////////////////// set caffe input
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

            datum.SerializeToString(&value);

            for (int i=1; i<=1; i++) {
               snprintf(key_cstr, kMaxKeyLength, "%08d", i);
               batch->Put(string(key_cstr), value);
            }
            db->Write(leveldb::WriteOptions(), batch);
            delete batch;
            batch = new leveldb::WriteBatch();
            ///////////////////////////// END set caffe input

            /////////////////////////////////////////////////////////// run deep learning CNN for one step to process the image
            const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);  

            const float* result_data = result[0]->cpu_data();

            if (shared->control==1) {

                shared->steerCmd = pow((result_data[0]-0.5)/0.4,3);

                if (result_data[1]>=0.5) {
                    shared->accelCmd = (result_data[1]-0.5)/0.4;
                    shared->brakeCmd = 0.0;
                } else {
                    shared->brakeCmd = -(result_data[1]-0.5)/0.4;
                    shared->accelCmd = 0.0;
                }
/*
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
*/
                printf("steerCmd: %f, accelCmd: %f, brakeCmd: %f\n\n", shared->steerCmd, shared->accelCmd, shared->brakeCmd);
                fflush(stdout);
            }

            shared->written=0;
        }  // end if (shared->written == 1)

        key=cvWaitKey( 1 );

        //////////////////////// Linux key
        if (key==1048603 || key==27) {  // Esc: exit
          shared->pause = 0;
          break;
        }
        else if (key==1048688 || key==112) shared->pause = 1-shared->pause;  // P: pause
        else if (key==1048675 || key==99) shared->control = 1-shared->control;  // C: autonomous driving on/off
        //////////////////////// END Linux key

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
        ////// END override drive by keyboard

    }  // end while (1) 

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvReleaseImage( &screenRGB );
    cvReleaseImage( &resizeRGB );
    ////////////////////// clean up opencv

    ////////////////////// clean up leveldb
    delete batch;
    delete db;
    ////////////////////// END clean up leveldb

    ////////////////////// clean up memory sharing
    if(shmdt(shm) == -1)  
    {  
        fprintf(stderr, "shmdt failed\n");  
        exit(EXIT_FAILURE);  
    }  
 
    if(shmctl(shmid, IPC_RMID, 0) == -1)  
    {  
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        exit(EXIT_FAILURE);  
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    exit(EXIT_SUCCESS); 
    ////////////////////// END clean up memory sharing 
}
