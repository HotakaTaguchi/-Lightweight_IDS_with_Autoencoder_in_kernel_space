#!/usr/bin/bash

BASEDIR=./saved_models
#backends=qnnpack
backends=fbgemm

#create model parameter(learning for float)
python3 main_create_params_float.py --save_dir ${BASEDIR}
#create model parameter(learning for qat)
python3 main_create_params_eBPF.py --save_dir ${BASEDIR} --backends ${backends}
#eBPF inference
sudo python3 main_inf_eBPF.py eno1
#user space inference
sudo python3 main_inf_float.py