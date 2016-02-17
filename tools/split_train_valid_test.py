### Splits the LevelDB into training, validation, and test sets.
### Also appends the marker value into the float_data of each used
### datum, which are read in from frames_all.txt

import caffe
import plyvel
from caffe.proto import caffe_pb2
import time

start = time.time()

with open('frames_all.txt', 'rb') as f:
    starting_frames = [int(l) for l in f.read().split('\n') if len(l) > 0]

db = plyvel.DB('/home/smile/edzhou/Thesis/data/TORCS_Training_1F')

# Split into train / valid / test, and also add marker data in.
db_train = plyvel.DB('/home/smile/edzhou/Thesis/data/TORCS_smooth_np_train',
                     create_if_missing=True,
                     error_if_exists=True)
db_valid = plyvel.DB('/home/smile/edzhou/Thesis/data/TORCS_smooth_np_valid',
                     create_if_missing=True,
                     error_if_exists=True)
db_test = plyvel.DB('/home/smile/edzhou/Thesis/data/TORCS_smooth_np_test',
                    create_if_missing=True,
                    error_if_exists=True)

datum = caffe_pb2.Datum()
datum_split = caffe_pb2.Datum()
N = 484815
# train_N = int(N * 0.2)
# valid_N = int(N * 0.4)
# assert (N - train_N - valid_N) == valid_N
train_N = 10000
valid_N = 10000
test_N = 5000

print ('''Train indices: [0, {0})\n'''
       '''Validation indices: [{0}, {1})\n'''
       '''Test indices: [{1}, {2})''').format(train_N,
                                              train_N + valid_N,
                                              train_N + valid_N + test_N)

# Initialize the write batches
wb_train = db_train.write_batch()
wb_valid = db_valid.write_batch()
wb_test = db_test.write_batch()

BATCH_SIZE = 100

for i, (key, value) in enumerate(db):
    
    # Parse leveldb value into Datum
    datum.ParseFromString(value)

    # Set new datum values
    datum_split.channels = datum.channels
    datum_split.height = datum.height
    datum_split.width = datum.width
    datum_split.data = datum.data                   # X
    for affordance_indicator in datum.float_data:   # y
        datum_split.float_data.append(affordance_indicator)   

    # Add the marker data in as well -- if the index is in the
    # i + 1 set read from file, it is from a different clip than
    # its predecessor image, and should have marker value 0.
    # Otherwise, it should have marker value 1.
    marker_value = 1
    if (i + 1) in starting_frames:
        marker_value = 0
    datum_split.float_data.append(marker_value)

    # Assign the data as train = [0, train_N),
    #                    valid = [train_N, train_N + valid_N)
    #                    test  = [train_N + valid_N, train_N + valid_N + test_N)

    if 0 <= i < train_N:
        # use same key as original db
        wb_train.put(key, datum_split.SerializeToString())

        # Every BATCH_SIZE values, write the batch and create a new one
        if i % BATCH_SIZE == 0 and i > 0: # Don't write the batch on the first one
            print 'Writing train batch, i = {0}'.format(i)
            wb_train.write()
            del wb_train
            wb_train = db_train.write_batch() # Re-initialize new batch

    elif train_N <= i < (train_N + valid_N):
        # If we just finished the train set, write the last train batch
        # before beginning the validation batches.
        if i == train_N:
            print 'Writing final train batch, i = {0}'.format(i)
            wb_train.write()
            del wb_train
        wb_valid.put(key, datum_split.SerializeToString())
        if i % BATCH_SIZE == 0:
            print 'Writing valid batch, i = {0}'.format(i)
            wb_valid.write()
            del wb_valid
            wb_valid = db_valid.write_batch()

    elif (train_N + valid_N) <= i < (train_N + valid_N + test_N):
        if i == (train_N + valid_N):
            print 'Writing final valid batch, i = {0}'.format(i)
            wb_valid.write()
            del wb_valid
        wb_test.put(key, datum_split.SerializeToString())
        if i % BATCH_SIZE == 0:
            print 'Writing test batch, i = {0}'.format(i)
            wb_test.write()
            del wb_test
            wb_test = db_test.write_batch()

    else:
        break

# Finish off writing the test batch.
print 'Writing final test batch, i = {0}'.format(i)
wb_test.write()
del wb_test

db.close()
db_train.close()
db_valid.close()
db_test.close()

end = time.time()
print 'Dataset partitioning took {0}'.format(end - start)