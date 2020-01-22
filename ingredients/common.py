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
from ingredients.common import *
from addict import Dict
from sacred import Ingredient

multi_label_ingredient = Ingredient("multi_label_ingredient")
nn_ingredient = Ingredient("nn_ingredient")
cnn_ingredient = Ingredient("cnn_ingredient", ingredients=[nn_ingredient, multi_label_ingredient])

@multi_label_ingredient.named_config
def use_session_dbmi():
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_sessions_seizure_data_4.pkl"
    session_instead_patient = True
    max_bckg_samps_per_file_test=None
    include_seizure_type = True
    max_bckg_samps_per_file = 100
    max_bckg_samps_per_file_test = -1


@multi_label_ingredient.named_config
def use_session_montage_dbmi():
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_sessions_montage_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_sessions_montage_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_sessions_montage_seizure_data_4.pkl"
    session_instead_patient = True
    max_bckg_samps_per_file_test=None
    include_montage_channels = True
    include_seizure_type = True
    max_bckg_samps_per_file = 100
    max_bckg_samps_per_file_test = -1

@cnn_ingredient.config
def cnn_ingredient_config():
    num_seconds = 4
    num_post_lin_h = 128
    use_inception = False
    use_batch_normalization = True
    use_lstm = False
    use_time_layers_first = True
    max_pool_size_time = (2,1)

    include_seizure_type = True
    attach_seizure_type_to_seizure_detect = False
    model_type = ""
    include_montage_channels = False
    attach_patient_layer_to_cnn_output = True
    max_pool_size = (2,2)
    max_pool_stride = (2,2)
    conv_spatial_filter=(3,3)
    conv_temporal_filter=(1,3)
    num_gpus=1
    num_conv_temporal_layers=1
    cnn_dropout = 0
    linear_dropout = 0.5
    lstm_h = 128
    lstm_return_sequence = False
    reduce_lr_on_plateau = False
    change_batch_size_over_time = None
    add_gaussian_noise = None
    pre_layer_h = 128
    num_lin_layer = 1
    num_layers = 3
    num_filters = 1
    num_temporal_filter=1
    num_post_cnn_layers = 2
    hyperopt_run = False
    make_model_in_parallel = False

@nn_ingredient.config
def nn_ingredient_config():
    def randomString(stringLength=16):
        """Generate a random string of fixed length """
        letters = string.ascii_uppercase
        return ''.join(random.choice(letters) for i in range(stringLength))
    model_name = randomString()
    optimizer_name="adam"
    patience=5
    early_stopping_on="val_loss"
    num_layers = 3
    num_gpus=1


@nn_ingredient.capture
def get_model_checkpoint(model_name, early_stopping_on):
    return ModelCheckpoint(model_name, monitor=early_stopping_on, save_best_only=True, verbose=1)


@nn_ingredient.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on)

@nn_ingredient.capture
def get_cb_list():
    return [get_model_checkpoint(), get_early_stopping()]


@multi_label_ingredient.capture
def getDataSampleGenerator(pre_cooldown, post_cooldown, sample_time, num_seconds, n_process):
    return er.EdfDatasetSegments(pre_cooldown=pre_cooldown, post_cooldown=post_cooldown, sample_time=sample_time, num_seconds=num_seconds, n_process=n_process)


@multi_label_ingredient.capture
def get_data(mode, max_samples, n_process, max_bckg_samps_per_file, num_seconds, max_bckg_samps_per_file_test, include_seizure_type, include_montage_channels, ref="01_tcp_ar", num_files=None):
    if max_bckg_samps_per_file_test is None:
        max_bckg_samps_per_file_test = max_bckg_samps_per_file
    if max_bckg_samps_per_file_test == -1:
        max_bckg_samps_per_file_test = None
    eds = getDataSampleGenerator()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()

    #increased n_process to deal with io processing

    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type, include_montage_channels=include_montage_channels)
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file_test, n_process=int(n_process), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type, include_montage_channels=include_montage_channels)
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file_test, n_process=int(n_process), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type, include_montage_channels=include_montage_channels)
    pkl.dump((train_edss, valid_edss, test_edss), open("/n/scratch2/ms994/seizure_multi_labels_edss_info.pkl", "wb"))
    multi_label_ingredient.add_artifact("/n/scratch2/ms994/seizure_multi_labels_edss_info.pkl")
    return train_edss, valid_edss, test_edss

@multi_label_ingredient.capture
def valid_dataset_class(balance_valid_dataset):
    if balance_valid_dataset:
        return RULDataGenMultipleLabels
    else:
        return DataGenMultipleLabels


@multi_label_ingredient.capture
def update_data(edss, seizure_classification_only, seizure_classes_to_use, include_seizure_type, include_montage_channels, remove_outlier_by_std_thresh, zero_center_each_channel, zero_out_patients=False):
    '''
    since we store the full string of the session or the patient instead of the index, we update the data to use the int index
    some of the tasks require different datasets and some filtering of the data i.e. only seizure classification or just some of the labels
    zero_out_patients: for use with valid and test set, since they shouldn't have patients from the train set and therefore predicting for them is wrong
    '''

    data = [datum[0] for datum in edss]
    if zero_out_patients:
        patient_labels = [0 for i in range(len(edss))]
    else:
        patients = [datum[1][1] for datum in edss]
        allPatient = list(set(patients))
        patient_labels = [allPatient.index(patient) for patient in patients]
    seizure_detection_labels = [datum[1][0] for datum in edss]
    if include_seizure_type:
        seizure_class_labels = [datum[1][2] for datum in edss]
    if include_montage_channels:
        montage_channel_labels = [datum[1][3] for datum in edss]
    keep_index = [True for i in range(len(data))]
    if seizure_classes_to_use is not None:
        for i, seizure_class_label in enumerate(seizure_class_labels):
            if seizure_detection_labels[i] and constants.SEIZURE_SUBTYPES[seizure_class_label] not in seizure_classes_to_use: #it's a seizure detection class and it should be excluded
                keep_index[i] = False
    if seizure_classification_only:
        for i, seizure_detect in enumerate(seizure_detection_labels):
            if not seizure_detect:
                keep_index[i] = False
    if remove_outlier_by_std_thresh is not None:
        removed = 0
        for i, datum in enumerate(data):
            if np.std(datum) > remove_outlier_by_std_thresh:
                keep_index[i] = False
                removed += 1
        print("We removed {} because unclean by std".format(removed))
    if zero_center_each_channel:
        for i, datum in enumerate(data):
            data[i] = datum - datum.mean(0)
    data_to_keep = []
    patient_labels_to_keep = []
    seizure_detect_to_keep = []
    seizure_class_labels_to_keep = []
    montage_channel_labels_to_keep = []

    for i, should_keep in enumerate(keep_index):
        if should_keep:
            data_to_keep.append(data[i])
            patient_labels_to_keep.append(patient_labels[i])
            seizure_detect_to_keep.append(seizure_detection_labels[i])
            if include_montage_channels:
                montage_channel_labels_to_keep.append(montage_channel_labels[i])
            if include_seizure_type:
                seizure_class_labels_to_keep.append(seizure_class_labels[i])
    if include_montage_channels and include_seizure_type:
        return [(data_to_keep[i], (seizure_detect_to_keep[i], patient_labels_to_keep[i], seizure_class_labels_to_keep[i], montage_channel_labels_to_keep[i].values)) for i in range(len(data_to_keep))], seizure_detect_to_keep, patient_labels_to_keep, seizure_class_labels_to_keep, montage_channel_labels_to_keep
    elif include_montage_channels:
        raise Exception("Not implemented yet")
    elif include_seizure_type:
        return [(data_to_keep[i], (seizure_detect_to_keep[i], patient_labels_to_keep[i], seizure_class_labels_to_keep[i])) for i in range(len(data_to_keep))], seizure_detect_to_keep, patient_labels_to_keep, seizure_class_labels_to_keep
    else:
        return [(data_to_keep[i], (seizure_detect_to_keep[i], patient_labels_to_keep[i])) for i in range(len(data_to_keep))], seizure_detect_to_keep, patient_labels_to_keep

@multi_label_ingredient.capture
def patient_func(tkn_file_paths, session_instead_patient):
    if session_instead_patient:
        return [read.parse_edf_token_path_structure(tkn_file_path)[1] + "/" + read.parse_edf_token_path_structure(tkn_file_path)[2] for tkn_file_path in tkn_file_paths]
    else:
        return [read.parse_edf_token_path_structure(tkn_file_path)[1] for tkn_file_path in tkn_file_paths]

@multi_label_ingredient.named_config
def debug():
    train_pkl = "/home/ms994/debug_train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/home/ms994/debug_valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/home/ms994/debug_test_multiple_labels_seizure_data_4.pkl"
    max_bckg_samps_per_file = 5 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = 5
    max_samples=10000
    include_seizure_type=True
    session_instead_patient = True

@multi_label_ingredient.config
def multi_label_ingredient_config():
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_sessions_montage_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_sessions_montage_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_sessions_montage_seizure_data_4.pkl"
    session_instead_patient = True
    lr = 0.005
    lr_decay = 0.75
    max_bckg_samps_per_file_test=None
    include_montage_channels = True
    include_seizure_type = True
    max_bckg_samps_per_file = 100
    max_bckg_samps_per_file_test = -1
    balance_valid_dataset = False

    patient_weight = 0
    seizure_weight = 1
    seizure_weight_decay = 1
    reduce_lr_on_plateau = False


    regenerate_data = False
    precache = True
    use_standard_scaler = False
    batch_size = 128
    pre_cooldown=4
    use_inception = False
    post_cooldown=None
    sample_time=4
    num_seconds=4
    n_process=20
    mode = er.EdfDatasetSegmentedSampler.DETECT_MODE
    n_process = 20
    include_seizure_type = False
    attach_seizure_type_to_seizure_detect = False
    seizure_classification_only = False
    seizure_classes_to_use = None
    remove_outlier_by_std_thresh = None
    zero_center_each_channel = False

@multi_label_ingredient.capture
def get_data_generators(train_pkl,  valid_pkl, test_pkl, regenerate_data, use_standard_scaler, precache, batch_size, n_process, include_seizure_type, include_montage_channels):
    allPatients = []
    seizureLabels = []
    validSeizureLabels = []
    validPatientInd = []
    patientInd = []
    if path.exists(train_pkl) and precache:
        print("Loading data")
        test_edss = pkl.load(open(test_pkl, 'rb'))
        print("loaded test")
        train_edss = pkl.load(open(train_pkl, 'rb'))
        print("loaded train")
        valid_edss = pkl.load(open(valid_pkl, 'rb'))
        print("Loading data completed")



        # validPatientInd

        seizureLabels = [datum[1][0] for datum in train_edss]
        if include_seizure_type:
            seizureSubtypes = [datum[1][2] for datum in train_edss]
            validSeizureSubtypes = [datum[1][2] for datum in valid_edss]
        validSeizureLabels = [datum[1][0] for datum in valid_edss]
    else:
        print("(Re)generating data")
        train_edss, valid_edss, test_edss = get_data()
        # tkn_file_paths = [train_edss.sampleInfo[key].token_file_path for key in train_edss.sampleInfo.keys()]

        # allPatients = list(set(patients))
        # patientInd = [allPatients.index(patient) for patient in patients]
        seizureLabels = [train_edss.sampleInfo[key].label for key in train_edss.sampleInfo.keys()]
        train_patients = patient_func( [train_edss.sampleInfo[key].token_file_path for key in train_edss.sampleInfo.keys()])
        validSeizureLabels = [valid_edss.sampleInfo[key].label for key in valid_edss.sampleInfo.keys()]
        valid_patients = patient_func( [valid_edss.sampleInfo[key].token_file_path for key in valid_edss.sampleInfo.keys()])
        testSeizureLabels = [test_edss.sampleInfo[key].label for key in test_edss.sampleInfo.keys()]
        test_patients = patient_func( [test_edss.sampleInfo[key].token_file_path for key in test_edss.sampleInfo.keys()])


        validPatientInd = [0 for i in range(len(validSeizureLabels))]
        if not include_seizure_type:
            for i in range(len(seizureLabels)):
                train_edss.sampleInfo[i].label = (seizureLabels[i], train_patients[i])
            for i in range(len(validSeizureLabels)):
                valid_edss.sampleInfo[i].label = (validSeizureLabels[i], valid_patients[i]) #the network has too many parameters if you include validation set patients (mutually exclusive) and the neural network should never choose validation patients anyways
        else:
            for i in range(len(seizureLabels)):
                train_edss.sampleInfo[i].label = (seizureLabels[i][0], train_patients[i], constants.SEIZURE_SUBTYPES.index(seizureLabels[i][1].lower()))
            for i in range(len(validSeizureLabels)):
                valid_edss.sampleInfo[i].label = (validSeizureLabels[i][0], valid_patients[i], constants.SEIZURE_SUBTYPES.index(validSeizureLabels[i][1].lower()))
            for i in range(len(testSeizureLabels)):
                test_edss.sampleInfo[i].label = (testSeizureLabels[i][0], test_patients[i], constants.SEIZURE_SUBTYPES.index(testSeizureLabels[i][1].lower()))

        train_edss = train_edss[:]
        valid_edss = valid_edss[:]
        test_edss = test_edss[:]



        pkl.dump(train_edss[:], open(train_pkl, 'wb'))
        pkl.dump(valid_edss[:], open(valid_pkl, 'wb'))
        pkl.dump(test_edss[:], open(test_pkl, 'wb'))

    #we want to have an actual string stored in record so we can do some more dissection on the segments, but we want an integer index when we run the code
    patients = [datum[1][1] for datum in train_edss]
    allPatients = list(set(patients))
    patientInd = [allPatients.index(patient) for patient in patients]
    validPatientInd = [0 for i in range(len(valid_edss))] #we don't actually care about predicting valid patients, since the split should be patient wise


    train_edss = update_data(train_edss)
    valid_edss = update_data(valid_edss, zero_out_patients=True)
    test_edss = update_data(test_edss, zero_out_patients=True)



    if include_seizure_type and not include_montage_channels:
        train_edss, seizureLabels, patientInd, seizureSubtypes = (train_edss)
        valid_edss, validSeizureLabels, validPatientInd, validSeizureSubtypes = (valid_edss)
        test_edss, _, _, testSeizureSubtypes = (test_edss)
    elif include_seizure_type and include_montage_channels:
        train_edss, seizureLabels, patientInd, seizureSubtypes, montageLabels = (train_edss)
        valid_edss, validSeizureLabels, validPatientInd, validSeizureSubtypes, validMontageLabels = (valid_edss)
        test_edss, testSeizureLabels, testPatientInd, testSeizureSubtypes, testMontageLabels = (test_edss)
    else:
        raise Exception("Not implemented yet")

    if use_standard_scaler:
        print("start standard scaling")
        # start = time()
        train_edss = read.EdfStandardScaler(
            train_edss, dataset_includes_label=True, n_process=n_process)
        train_edss.use_mp=False
        # print(time-start)
        valid_edss = read.EdfStandardScaler(
            valid_edss, dataset_includes_label=True, n_process=n_process)
        valid_edss.use_mp=False

        # print(time-start)

        test_edss = read.EdfStandardScaler(
            test_edss, dataset_includes_label=True, n_process=n_process)
        test_edss.use_mp=False

        # print(time-start)

        print("completed")


    if include_seizure_type and not include_montage_channels:
        edg = RULDataGenMultipleLabels(train_edss, num_labels=3, precache=not use_standard_scaler, batch_size=batch_size, labels=[seizureLabels, patientInd, seizureSubtypes], n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)),)
        valid_edg = valid_dataset_class()(valid_edss, num_labels=3, precache=not use_standard_scaler, batch_size=batch_size*4, labels=[validSeizureLabels, validPatientInd, validSeizureSubtypes], xy_tuple_form=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), shuffle=False)
        test_edg = DataGenMultipleLabels(test_edss, num_labels=3, precache=not use_standard_scaler, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), batch_size=batch_size*4, shuffle=False)
    elif include_seizure_type and include_montage_channels:
        edg = RULDataGenMultipleLabels(train_edss, num_labels=4, precache=not use_standard_scaler, class_type=["nominal", "nominal", "nominal", "quantile"], batch_size=batch_size, labels=[seizureLabels, patientInd, seizureSubtypes, montageLabels], n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES), len(constants.MONTAGE_COLUMNS)),)
        valid_edg = valid_dataset_class()(valid_edss, num_labels=4, precache=not use_standard_scaler, class_type=["nominal", "nominal", "nominal", "quantile"], batch_size=batch_size*4, labels=[validSeizureLabels, validPatientInd, validSeizureSubtypes, validMontageLabels], xy_tuple_form=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES), len(constants.MONTAGE_COLUMNS)), shuffle=False)
        test_edg = DataGenMultipleLabels(test_edss, num_labels=4, precache=not use_standard_scaler, class_type=["nominal", "nominal", "nominal", "quantile"], labels=[testSeizureLabels, testPatientInd, testSeizureSubtypes, testMontageLabels], n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES), len(constants.MONTAGE_COLUMNS)), batch_size=batch_size*4, shuffle=False)
    elif not include_seizure_type and not include_montage_channels:
        edg = RULDataGenMultipleLabels(train_edss, num_labels=2, precache=not use_standard_scaler, labels=[seizureLabels, patientInd], batch_size=batch_size, n_classes=(2, len(allPatients)),) #learning means we are more likely to be affected by batch size, both for OOM in gpu and as a hyperparamer
        valid_edg = valid_dataset_class()(valid_edss, num_labels=2, precache=not use_standard_scaler, labels=[validSeizureLabels, validPatientInd], batch_size=batch_size*4, xy_tuple_form=True, n_classes=(2, len(allPatients)), shuffle=False) #batch size doesn't matter as much when we aren't learning but we still need batches to avoid OOM
        if len(test_edss[0][1]) > 1: #we throw out the seizure type label
            data = [datum[0] for datum in test_edss]
            labels = [datum[1][0] for datum in test_edss]
            test_edss = [(data[i], labels[i]) for i in range(len(data))]
        test_edg = EdfDataGenerator(test_edss, n_classes=2, precache=not use_standard_scaler, batch_size=batch_size, shuffle=False)
    return edg, valid_edg, test_edg, len(allPatients)




@nn_ingredient.capture
def get_optimizer(optimizer_name):
    if optimizer_name == "adam":
        return optimizers.Adam
    elif optimizer_name == "sgd":
        return optimizers.SGD

@cnn_ingredient.capture
def get_model(
    num_patients,
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
    max_pool_size_time,
    patient_weight,
    seizure_weight,
    include_seizure_type,
    attach_seizure_type_to_seizure_detect,
    lstm_h,
    lstm_return_sequence,
    model_type,
    add_gaussian_noise,
    include_montage_channels,
    attach_patient_layer_to_cnn_output):
    input_time_size = num_seconds * constants.COMMON_FREQ
    x = Input((input_time_size, 21, 1)) #time, ecg channel, cnn channel
    if add_gaussian_noise is not None:
        y = GaussianNoise(add_gaussian_noise)(x)
    else:
        y = x
    if num_lin_layer != 0:
        y = Reshape((input_time_size, 21))(y) #remove channel dim
        for i in range(num_lin_layer):
            y = TimeDistributed(Dense(pre_layer_h, activation="relu"))(y)
            # y = TimeDistributed(Dropout(linear_dropout))(y)

        y = Reshape((input_time_size, pre_layer_h, 1))(y) #add back in channel dim
    else:
        y = x
    if use_inception:
        _, y = inception_like_pre_layers(input_shape=(input_time_size,21,1), x=y, dropout=cnn_dropout, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride, num_layers=num_layers, num_filters=num_filters)
    elif model_type=="time_distributed_dense":
        for i in range(num_conv_temporal_layers):
            y = Conv2D(num_conv_temporal_layers, conv_temporal_filter, activation="relu")(y)
            y = MaxPool2D(max_pool_size)(y)

        for i in range(2):
            if use_batch_normalization:
                y = layers.BatchNormalization()(y)
            y = Conv2D(num_filters, conv_spatial_filter, activation="relu")(y)
            y = MaxPool2D(max_pool_size)(y)
            y = TimeDistributed(Dense(y.get_shape()[2].value, activation="relu"))(y)
            y = TimeDistributed(Dropout(cnn_dropout))(y)
            y = TimeDistributed(Dense(y.get_shape()[2].value, activation="relu"))(y)
    elif model_type=="cnn1d":
        y = layers.Reshape((input_time_size, 21))(y)
        for i in range(num_layers):
            if use_batch_normalization:
                y = layers.BatchNormalization()(y)
            y = layers.Conv1D(num_filters, (4), activation="relu")(y)
            y = layers.MaxPool1D((2))(y)

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
    if not use_lstm:
        y_flatten = Flatten()(y)
        y = y_flatten
    else:
        y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)
        y = CuDNNLSTM(lstm_h, return_sequences=lstm_return_sequence)(y)
        if lstm_return_sequence:
            y = Flatten(y)


    for i in range(num_post_cnn_layers):
        y = Dense(num_post_lin_h, activation='relu')(y)
        y = Dropout(linear_dropout)(y)

    y_seizure_subtype = Dense(len(constants.SEIZURE_SUBTYPES), activation="softmax", name="seizure_subtype")(y)
    if include_seizure_type and attach_seizure_type_to_seizure_detect:
        y = Concatenate()([y, y_seizure_subtype])
    y_seizure = Dense(2, activation="softmax", name="seizure")(y)
    if not attach_patient_layer_to_cnn_output:
        y_patient = Dense(num_patients, activation="softmax", name="patient")(y)
    else:
        y_patient = Dense(num_patients, activation="softmax", name="patient")(y_flatten)
    y_montage_channel = Dense(len(constants.MONTAGE_COLUMNS), activation="sigmoid", name="montage_channel")(y)



    seizure_model = Model(inputs=x, outputs=[y_seizure])

    if include_seizure_type and include_montage_channels:
        seizure_patient_model = Model(inputs=[x], outputs=[y_seizure, y_patient,  y_seizure_subtype, y_montage_channel])
        val_train_model = Model(inputs=x, outputs=[y_seizure, y_seizure_subtype, y_montage_channel])
    elif include_seizure_type:
        seizure_patient_model = Model(inputs=[x], outputs=[y_seizure, y_patient,  y_seizure_subtype,])
        val_train_model = Model(inputs=x, outputs=[y_seizure, y_seizure_subtype])
    else:
        seizure_patient_model = Model(inputs=[x], outputs=[y_seizure, y_patient,])
        val_train_model = seizure_model

    patient_model = Model(inputs=[x], outputs=[y_patient])
    print(seizure_patient_model.summary())
    if num_gpus > 1:
        seizure_model = multi_gpu_model(seizure_model, num_gpus)
        seizure_patient_model = multi_gpu_model(seizure_patient_model, num_gpus)
        patient_model = multi_gpu_model(patient_model, num_gpus)

    seizure_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy"], metrics=["binary_accuracy"])
    if include_seizure_type and include_montage_channels:
        loss_weights = [K.variable(seizure_weight),K.variable(patient_weight), K.variable(seizure_weight), K.variable(seizure_weight)]
        seizure_patient_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "binary_crossentropy"], loss_weights=loss_weights, metrics=["categorical_accuracy", f1])
    elif include_seizure_type:
        loss_weights = [K.variable(seizure_weight),K.variable(patient_weight), K.variable(seizure_weight)]
        seizure_patient_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"], loss_weights=loss_weights, metrics=["categorical_accuracy", f1])
    elif not include_seizure_type and not include_montage_channels:
        loss_weights = [K.variable(seizure_weight),K.variable(patient_weight)]
        seizure_patient_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy",  "categorical_crossentropy"], loss_weights=loss_weights, metrics=["categorical_accuracy", f1])
    else:
        raise Exception("Not Implemented")

    patient_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
    return seizure_model, seizure_patient_model, patient_model, val_train_model, x, y, loss_weights

global_model = None

@multi_label_ingredient.capture
def recompile_model(seizure_patient_model, epoch_num, seizure_weight, min_seizure_weight, patient_weight,  loss_weights, include_seizure_type, lr, lr_decay, seizure_weight_decay, reduce_lr_on_plateau, include_montage_channels):
    if seizure_weight_decay is not None:
        if seizure_weight_decay is None:
            seizure_weight_decay = 1
        if lr_decay == 0 or lr_decay is None:
            new_lr = lr
        elif not reduce_lr_on_plateau:
            new_lr = lr * (lr_decay) ** ((epoch_num))
        else:
            new_lr = lr
        # weight_decay = seizure_weight_decay ** ((epoch_num))
        # new_weight = seizure_weight * weight_decay
        if min_seizure_weight is None:
            min_seizure_weight = 0
        if min_seizure_weight is not None or min_seizure_weight != 0:
            new_weight = (seizure_weight - min_seizure_weight) * np.e ** (np.log(seizure_weight_decay) * epoch_num + 1) + min_seizure_weight * np.e
            new_weight /= np.e
        print("Epoch: {}, Seizure Weight: {}, Patient Weight: {}, lr: {}".format(epoch_num, new_weight, patient_weight, new_lr))
        K.set_value(seizure_patient_model.optimizer.lr, lr)
        if include_seizure_type and include_montage_channels:
            seizure_patient_model.compile(seizure_patient_model.optimizer, loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "binary_crossentropy"], loss_weights=[new_weight, patient_weight, new_weight, new_weight], metrics=["categorical_accuracy", f1])

        elif include_seizure_type and seizure_weight_decay is not None:
             # K.set_value(
             #Don't throw away old optimizer TODO: check and see if adam keeps any state in its optimizer object
            seizure_patient_model.compile(seizure_patient_model.optimizer, loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"], loss_weights=[new_weight, patient_weight, new_weight], metrics=["categorical_accuracy", f1])
        elif seizure_weight_decay is not None:
            seizure_patient_model.compile(seizure_patient_model.optimizer, loss=["categorical_crossentropy",  "categorical_crossentropy"], loss_weights=[new_weight, patient_weight,], metrics=["categorical_accuracy", f1])
    # seizure_patient_model.metrics_tensors += seizure_patient_model.outputs #grab output!

    return seizure_patient_model
