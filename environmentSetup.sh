#!/usr/bin/env bash

conda env create -f env.yaml
source activate keras
pip install git+https://github.com/farizrahman4u/recurrentshop.git #3d08db80a154fbcf3730f7c7f35d6f7973b4abbc
pip install git+https://github.com/farizrahman4u/seq2seq.git #c020ccfc1fa3a651be272f8b4be48a10f9c3f0fa
