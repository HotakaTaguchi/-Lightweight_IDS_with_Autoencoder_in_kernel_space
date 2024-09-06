#!/usr/bin/bash

BASEDIR=./saved_models
#backends=qnnpack
backends=fbgemm

python3 pytorch_encoder_c_params.py --save_dir ${BASEDIR} --backends ${backends}
make
python3 pytorch_encoder_c.py --save_dir ${BASEDIR}