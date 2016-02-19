Note: There are 5 model snapshots in this directory -- the 3 models with
2_ in the name are the result of the second training session, which was
initialized at the 20k iteration.  Therefore, we have

10000 iterations ==> parametric_init_iter_10000.*
20000 iterations ==> parametric_init_iter_20000.*
30000 iterations ==> parametric_init_2__iter_10000.*
40000 iterations ==> parametric_init_2__iter_20000.*
50000 iterations ==> parametric_init_2__iter_30000.*

and we use the last model as the initialization to our nonparametric net.