import numpy as np
import caffe

caffe.set_mode_gpu()
net = caffe.Net('nonparametric/lstm_np_01/train_valid.prototxt',
                'nonparametric/lstm_np_01/np_01_iter_10000.caffemodel',
                caffe.TEST)
beta = net.params['nprelu1'][0].data