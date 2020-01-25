from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
from keras import backend as K

# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix, mean_squared_error
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels, RULEdfDataGenerator, RULDataGenMultipleLabels
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like_pre_layers, conv2d_gridsearch_pre_layers
from keras import optimizers
from keras.layers import Dense, TimeDistributed, Input, Reshape, Dropout, LSTM, Flatten, Concatenate, CuDNNLSTM, GaussianNoise, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, TimeDistributed, Dense
import keras.layers as layers
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
import pickle as pkl
import sacred
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model
from keras_models import train
import constants
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras_models.metrics import f1
import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import multi_gpu_model
from time import time

from addict import Dict
from predictSeizureMultipleLabels import *
ex_lstm = sacred.Experiment(name="seizure_LSTM_exp", ingredients=[ex])

# ex_lstm.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

@ex_lstm.config
def lstm_config():
    a = ""


@ex_lstm.capture
def lstm_model(input_time_size):
    x = Input((input_time_size, 21, 1)) #time, ecg channel, cnn channel

@ex_lstm.main
def main():
    print("hello world")
    edg, valid_edg, test_edg, len_all_patients = get_data_generators()
    model = lstm_model()


if __name__ == "__main__":
    ex_lstm.run_commandline()
