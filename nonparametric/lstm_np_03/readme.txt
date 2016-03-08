ReLU layers replaced with NonParaReLU (sin/cos), initialized from 50k iteration model
(30k _2) in lstm_np_para_init, run for 50k iterations.

./build/tools/caffe train -solver nonparametric/lstm_np_03/solver.prototxt 2>&1 -weights nonparametric/lstm_np_para_init/parametric_init_2__iter_30000.caffemodel | tee nonparametric/lstm_np_03/lstm_np_03_log.txt
