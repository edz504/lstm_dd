import numpy as np
import caffe

caffe.set_mode_gpu()
net = caffe.Net('nonparametric/lstm_np_03/train_valid.prototxt',
                'nonparametric/lstm_np_03/np_03_iter_10000.caffemodel',
                caffe.TEST)

# net = caffe.Net('nonparametric/lstm_np_03/train_valid.prototxt',
#                 'nonparametric/lstm_np_03/np_03_iter_50000.caffemodel',
#                 caffe.TEST)

beta_sin = net.params['nprelu1'][0].data
beta_cos = net.params['nprelu1'][1].data

# without updates, beta_sin should be all zeroes (initialization)