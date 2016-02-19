### Checks the training, validation, and test sets to ensure they
### were split into the right dimensions.

import caffe
import plyvel
from caffe.proto import caffe_pb2
import numpy as np
import time

db_train = plyvel.DB('data/TORCS_smooth_np_train')
db_valid = plyvel.DB('data/TORCS_smooth_np_valid')
db_test = plyvel.DB('data/TORCS_smooth_np_test')

datum = caffe_pb2.Datum()

y_size_vec = []
X_size_vec = []
for i, (key, value) in enumerate(db_train):
    datum.ParseFromString(value)
    y_size_vec.append(len(datum.float_data))
    X_size_vec.append(len(datum.data))
    if i > 1000:
        break
print np.mean(y_size_vec)
print np.mean(X_size_vec)

y_size_vec = []
X_size_vec = []
for i, (key, value) in enumerate(db_valid):
    datum.ParseFromString(value)
    y_size_vec.append(len(datum.float_data))
    X_size_vec.append(len(datum.data))
    if i > 1000:
        break
print np.mean(y_size_vec)
print np.mean(X_size_vec)

y_size_vec = []
X_size_vec = []
for i, (key, value) in enumerate(db_test):
    datum.ParseFromString(value)
    y_size_vec.append(len(datum.float_data))
    X_size_vec.append(len(datum.data))
    if i > 1000:
        break
print np.mean(y_size_vec)
print np.mean(X_size_vec)