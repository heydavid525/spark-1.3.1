#!/usr/bin/env python

import os
from os.path import dirname
from os.path import join
import time

main_ip = "10.54.1.36"
data_path = "/users/wdai/datasets/lr/synth/lr2sp_dim500_s5000_nnz200x1.libsvm.0"
#data_path = "/proj/BigLearning/jinlianw/data/criteo_click/libsvm/day_0/data.libsvm.0"
#data_path = "/l0/criteo"
#data_path = "/users/wdai/datasets/lr/criteo/day_0.3"
#data_path = \
#  '\'/proj/BigLearning/jinlianw/data/criteo_click/libsvm/day_0/data.libsvm.[0-3]\''
#data_path = "/users/wdai/datasets/lr/criteo"
driver_memory = "32g"

num_iterations = 10
step_size = 1e-7
reg_type = "NONE" # Options: L1, L2, NONE
reg_lambda = 0
data_format = "LibSVM"
minibatch_fraction = 1
num_legs = 1
num_dups = 1
num_parallelism = 48

script_dir = dirname(os.path.realpath(__file__))
spark_dir = dirname(script_dir)

cmd = "time %s/bin/spark-submit" % spark_dir
cmd += " --class LogisticRegression"
cmd += " --main spark://%s:7077" % main_ip
cmd += " --driver-memory " + driver_memory
cmd += " %s/lr/target/lr-1.0.jar" % script_dir
cmd += " --numIterations " + str(num_iterations)
cmd += " --stepSize " + str(step_size)
cmd += " --regType " + reg_type
cmd += " --regParam " + str(reg_lambda)
cmd += " --minibatchFraction " + str(minibatch_fraction)
cmd += " --dataFormat " + data_format
cmd += " --numLegs " + str(num_legs)
cmd += " --numDups " + str(num_dups)
cmd += " --numParallelism " + str(num_parallelism)
cmd += " " + data_path

print cmd
os.system(cmd)
