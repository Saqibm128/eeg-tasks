import sys, os
sys.path.append(os.path.realpath(".."))
from sacred.observers import MongoObserver
try: import cPickle as pkl
except: import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
import data_reader as read
# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels, RULEdfDataGenerator
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like_pre_layers, conv2d_gridsearch_pre_layers
from keras import optimizers
from keras.layers import Dense, TimeDistributed, Input, Reshape, Dropout, LSTM, Flatten
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
import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model

from addict import Dict
ex = sacred.Experiment(name="seizure_conv_exp")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@ex.named_config
def no_lin_pre_layer():
    num_lin_layer = 0

@ex.named_config
def no_stride_channels():
    '''
    Don't stride on channels
    '''
    max_pool_stride = (1,1)

@ex.config
def config():
    model_name = "/n/scratch2/ms994/out/" + randomString() + ".h5" #set to rando string so we don't have to worry about collisions
    mode=er.EdfDatasetSegmentedSampler.DETECT_MODE
    max_samples=None
    max_pool_size = (2,2)
    max_pool_stride = (2,2)
    steps_per_epoch = None

    conv_spatial_filter=(3,3)
    conv_temporal_filter=(1,3)
    num_gpus=1
    num_conv_temporal_layers=1

    imbalanced_resampler = "rul"
    pre_cooldown=4
    use_inception = False
    post_cooldown=None
    sample_time=4
    num_seconds=4
    n_process=20
    mode = er.EdfDatasetSegmentedSampler.DETECT_MODE
    cnn_dropout = 0
    linear_dropout = 0.5


    precache = True
    regenerate_data = False
    train_pkl = "/n/scratch2/ms994/train_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_seizure_data_4.pkl"
    batch_size = 32

    pre_layer_h = 100
    num_lin_layer = 2

    patience=5
    early_stopping_on="val_binary_accuracy"
    fit_generator_verbosity = 2
    num_layers = 3
    num_filters = 1
    num_temporal_filter=1
    num_post_cnn_layers = 2
    hyperopt_run = False
    make_model_in_parallel = False
    randomly_reorder_channels = False #use if we want to try to mess around with EEG order
    random_channel_ordering = get_random_channel_ordering()

    num_post_lin_h = 5

    use_batch_normalization = True

    max_bckg_samps_per_file = 100
    max_samples=None
    use_standard_scaler = False
    use_filtering = True
    ref = "01_tcp_ar"
    combined_split = None
    lr = 0.001

    use_lstm = False
    use_time_layers_first = False
    max_pool_size_time = None

    epochs=100

@ex.capture
def getImbResampler(imbalanced_resampler):
    if imbalanced_resampler is None:
        return None
    elif imbalanced_resampler == "SMOTE":
        return SMOTE()
    elif imbalanced_resampler == "rul":
        return RandomUnderSampler()


def get_random_channel_ordering():
    channel_ordering = [i for i in range(len(util_funcs.get_common_channel_names()))]
    np.random.shuffle(channel_ordering)
    return channel_ordering

@ex.capture
def resample_x_y(x, y, imbalanced_resampler):
    if imbalanced_resampler is None:
        return x, y
    else:
        oldShape = x.shape
        resampleX, resampleY = getImbResampler().fit_resample(x.reshape(x.shape[0], -1), y)
        return resampleX.reshape(resampleX.shape[0], *oldShape[1:]), resampleY


@ex.capture
def getDataSampleGenerator(pre_cooldown, post_cooldown, sample_time, num_seconds, n_process):
    return er.EdfDatasetSegments(pre_cooldown=pre_cooldown, post_cooldown=post_cooldown, sample_time=sample_time, num_seconds=num_seconds, n_process=n_process)


@ex.capture
def get_data(mode, max_samples, n_process, max_bckg_samps_per_file, num_seconds, ref="01_tcp_ar", num_files=None):
    eds = getDataSampleGenerator()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()

    #increased n_process to deal with io processing
    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1))
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1))
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1))
    return train_edss, valid_edss, test_edss

@ex.capture
def get_model(
    num_seconds,
    lr,
    pre_layer_h,
    num_lin_layer,
    num_post_cnn_layers,
    num_post_lin_h,
    num_layers,
    num_filters,
    max_pool_stride,
    use_inception,
    cnn_dropout,
    linear_dropout,
    num_gpus,
    max_pool_size,
    conv_spatial_filter,
    conv_temporal_filter,
    num_conv_temporal_layers,
    num_temporal_filter,
    use_batch_normalization,
    use_lstm,
    use_time_layers_first,
    max_pool_size_time):
    input_time_size = num_seconds * constants.COMMON_FREQ
    x = Input((input_time_size, 21, 1)) #time, ecg channel, cnn channel
    if num_lin_layer != 0:
        y = Reshape((input_time_size, 21))(x) #remove channel dim



        for i in range(num_lin_layer):
            y = TimeDistributed(Dense(pre_layer_h, activation="relu"))(y)
            y = TimeDistributed(Dropout(linear_dropout))(y)

        y = Reshape((input_time_size, pre_layer_h, 1))(y) #add back in channel dim
    else:
        y = x
    if use_inception:
        _, y = inception_like_pre_layers(input_shape=(input_time_size,21,1), x=y, dropout=cnn_dropout, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride, num_layers=num_layers, num_filters=num_filters)
    else:
        _, y = conv2d_gridsearch_pre_layers(input_shape=(input_time_size,21,1),
                                            x=y,
                                            conv_spatial_filter=conv_spatial_filter,
                                            conv_temporal_filter=conv_temporal_filter,
                                            num_conv_temporal_layers=num_conv_temporal_layers,
                                            max_pool_size=max_pool_size,
                                            max_pool_stride=max_pool_stride,
                                            dropout=cnn_dropout,
                                            num_conv_spatial_layers=num_layers,
                                            num_spatial_filter=num_filters,
                                            num_temporal_filter=num_temporal_filter,
                                            use_batch_normalization=use_batch_normalization,
                                            max_pool_size_time=max_pool_size_time,
                                            time_convolutions_first=use_time_layers_first)
    # y = Dropout(0.5)(y)
    for i in range(num_post_cnn_layers):
        y = Dense(num_post_lin_h, activation='relu')(y)
        y = Dropout(linear_dropout)(y)
    if not use_lstm:
        y = Flatten()(y)
        y_seizure = Dense(2, activation="softmax", name="seizure")(y)
    else:
        y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)
        y_seizure = LSTM(2)(y)
    model = Model(inputs=x, outputs=[y_seizure])
    print(model.summary())
    if num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)
    model.compile(optimizers.Adam(lr=lr), loss=["binary_crossentropy"], metrics=["binary_accuracy"])

    return model

global_model = None

@ex.capture
def set_global_model():
    global global_model
    global_model = get_model()

@ex.capture
def get_model_checkpoint(model_name, early_stopping_on):
    return ModelCheckpoint(model_name, monitor=early_stopping_on, save_best_only=True, verbose=1)


@ex.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on)

@ex.capture
def get_cb_list():
    return [get_model_checkpoint(), get_early_stopping()]

@ex.capture
def reorder_channels(data, randomly_reorder_channels, random_channel_ordering):
    if randomly_reorder_channels:
        newData = []
        for datum_pair in data:
            datum_pair_first = datum_pair[0][:,random_channel_ordering]
            newData.append((datum_pair_first, datum_pair[1]))
        return newData
    else:
        return data

@ex.capture
@functools.lru_cache(10)
def get_data_generators(train_pkl,  valid_pkl, test_pkl, regenerate_data, use_standard_scaler, precache, batch_size, n_process):
    if path.exists(train_pkl) and precache:
        test_edss = pkl.load(open(test_pkl, 'rb'))
        train_edss = pkl.load(open(train_pkl, 'rb'))
        valid_edss = pkl.load(open(valid_pkl, 'rb'))
    else:
        train_edss, valid_edss, test_edss = get_data()
        train_edss = train_edss[:]
        valid_edss = valid_edss[:]
        test_edss = test_edss[:]


        pkl.dump(train_edss[:], open(train_pkl, 'wb'))
        pkl.dump(valid_edss[:], open(valid_pkl, 'wb'))
        pkl.dump(test_edss[:], open(test_pkl, 'wb'))




    if use_standard_scaler:
        train_edss = read.EdfStandardScaler(
            train_edss, dataset_includes_label=True, n_process=n_process)[:]
        valid_edss = read.EdfStandardScaler(
            valid_edss, dataset_includes_label=True, n_process=n_process)[:]
        test_edss = read.EdfStandardScaler(
            test_edss, dataset_includes_label=True, n_process=n_process)[:]

    def split_tuples(data):
        return np.stack([datum[0] for datum in data]), np.stack([datum[1] for datum in data])

    trainData, trainLabels = split_tuples(train_edss[:])
    validData, validLabels = split_tuples(valid_edss[:])
    trainDataResampled, trainLabelsResampled = resample_x_y(trainData, trainLabels)
    validDataResampled, validLabelsResampled = resample_x_y(validData, validLabels)
    train_edss_resampled = list(zip(trainDataResampled, trainLabelsResampled))
    valid_edss_resampled = list(zip(validDataResampled, validLabelsResampled))

    train_edss_resampled = reorder_channels(train_edss_resampled)
    valid_edss_resampled = reorder_channels(valid_edss_resampled)
    test_edss = reorder_channels(test_edss)

    if regenerate_data:
        return
    edg = EdfDataGenerator(train_edss_resampled, n_classes=2, precache=True, batch_size=batch_size)
    valid_edg = EdfDataGenerator(valid_edss_resampled, n_classes=2, precache=True, batch_size=batch_size)
    test_edg = EdfDataGenerator(test_edss[:], n_classes=2, precache=True, batch_size=batch_size, shuffle=False)
    return edg, valid_edg, test_edg
@ex.main
def main(model_name, mode, num_seconds, imbalanced_resampler,  regenerate_data, epochs, fit_generator_verbosity, batch_size, n_process, steps_per_epoch):
    model = get_model() #get the model first to see if the parameters passed in fail (negative dim size)

    edg, valid_edg, test_edg = get_data_generators()
    if regenerate_data:
        return

    if steps_per_epoch is None:
        history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs)
    else:
        history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs, steps_per_epoch=steps_per_epoch)

    model = load_model(model_name)

    y_pred = model.predict_generator(test_edg)

    results = Dict()
    results.history = history.history
    results.seizure.acc = accuracy_score(y_pred.argmax(1), np.array([data[1] for data in test_edg.dataset]).astype(int))
    results.seizure.f1 = f1_score(y_pred.argmax(1), np.array([data[1] for data in test_edg.dataset]).astype(int))
    results.seizure.classification_report = classification_report(np.array([data[1] for data in test_edg.dataset]).astype(int), y_pred.argmax(1), output_dict=True),
    try:
        results.seizure.AUC = roc_auc_score(y_pred.argmax(1), np.array([data[1] for data in test_edg.dataset]).astype(int))
    except Exception:
        results.seizure.AUC = "failed to calculate"

    return results.to_dict()


if __name__ == "__main__":
    ex.run_commandline()
