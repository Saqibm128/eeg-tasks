from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
import data_reader as read
from keras import backend as K

# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from keras_models.homeoschedastic import HomeoschedasticMultiLossLayer, RelativeHomeoschedasticMultiLossLayer
from keras.losses import categorical_crossentropy, binary_crossentropy

from addict import Dict
ex = sacred.Experiment(name="seizure_long_term")
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))
import preprocessingV2.preprocessingV2 as ppv2

@ex.config
def config():
    batch_size=16
    filter_size=(3,3)
    max_pool_size = (1,2)
    num_filters=1
    num_layers=6
    lstm_h=32
    post_lin_h =32
    num_lin_layers=2
    gaussian_noise=2
    dropout = 0.5
    num_workers = 8
    num_epochs=100
    max_queue_size = 30

@ex.capture
def getCachedData():
    testDR = ppv2.FileDataReader(split="test", directory="/n/scratch2/ms994/medium_size/test", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/test/20sIndex.pkl", "rb")))
    trainDR = ppv2.RULDataReader(split="train", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/train/20sIndex.pkl", "rb")))
    validDR = ppv2.FileDataReader(split="valid", directory="/n/scratch2/ms994/medium_size/valid", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/valid/20sIndex.pkl", "rb")))
    return trainDR, validDR, testDR

@ex.capture
def get_dg(batch_size):
    train, valid, test = getCachedData()
    return EdfDataGenerator(train, precache=False, batch_size=batch_size, n_classes=2), EdfDataGenerator(valid, precache=False, batch_size=batch_size, n_classes=2), EdfDataGenerator(test, precache=False, batch_size=batch_size, n_classes=2)

@ex.capture
def get_model(num_filters, filter_size, gaussian_noise, num_layers, max_pool_size, lstm_h, num_lin_layers, post_lin_h, dropout):
    input = Input((11,21,1000,1))
    x = GaussianNoise(gaussian_noise)(input)
    for i in range(num_layers):
        x = BatchNormalization()(x)
        x = TimeDistributed(Conv2D(num_filters, filter_size, activation="relu"))(x)
        x = TimeDistributed(MaxPool2D(max_pool_size))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(lstm_h, activation="relu", return_sequences=True)(x)
    for i in range(num_lin_layers):
        x = TimeDistributed(Dense(post_lin_h, activation="relu"))(x)
        x = TimeDistributed(Dropout(dropout))(x)

    y = TimeDistributed(Dense(2, activation="relu"))(x)
    model = Model(inputs=[input], outputs=[y])
    model.compile(keras.optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["binary_accuracy"])
    return model


@ex.main
def main(num_workers, max_queue_size, num_epochs):
    valid, train, test = get_dg()
    model = get_model()
    model.summary()
    data = train[0]
    model.fit_generator( \
                        train, \
                        validation_data=valid, \
                        steps_per_epoch=1,\
                        epochs=num_epochs, \
                        workers=num_workers, \
                        max_queue_size=max_queue_size, \
                        use_multiprocessing=True, \
                        callbacks=[ \
                                   keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr*0.9, verbose=1), \
                                   keras.callbacks.EarlyStopping(patience=20), \
                                   keras.callbacks.ModelCheckpoint("bestModel.h5", save_best_only=True), \
                                   ])
    bestModel = keras.models.load_model("bestModel.h5")
    test.batch_size = 32
    predictions = bestModel.predict_generator(test, )


    raise Exception()
    print("hi")

if __name__ == "__main__":
    ex.run_commandline()
