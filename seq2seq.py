import sys, os
sys.path.append(os.path.realpath("/home/ms994/miniconda3/envs/keras-redo/lib/python3.7/site-packages"))
sys.path.append(os.path.realpath("../"))

import util_funcs
from importlib import reload
reload(util_funcs)
from copy import deepcopy as cp

import data_reader as read
import pandas as pd


reload(read)

edfRawData = read.EdfDataset("train", "01_tcp_ar", num_files=1, n_process=7)
edfFFTData = read.EdfFFTDatasetTransformer(edfRawData, window_size=pd.Timedelta(seconds=1), precache=False, n_process=7)

fftData = edfFFTData[0:1]
