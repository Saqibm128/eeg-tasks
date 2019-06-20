#!/usr/bin/env bash

source activate keras && python script_runner.py ../train_01_tcp_ar_fft.pkl --num_process 8 --dim_red ica
