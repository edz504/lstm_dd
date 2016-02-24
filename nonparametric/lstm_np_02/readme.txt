ReLU layers replaced with NonParaReLU, initialized from 20k iteration model
in lstm_np_para_init, run for 20k iterations.

./build/tools/caffe train -solver nonparametric/lstm_np_02/solver.prototxt 2>&1 -weights nonparametric/lstm_np_para_init/parametric_init_iter_20000.caffemodel | tee nonparametric/lstm_np_02/lstm_np_02_log.txt
