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

from addict import Dict
ex = sacred.Experiment(name="seizure_hand_eng_fft_coh_exp")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@ex.capture
def getDataSampleGenerator(pre_cooldown, post_cooldown, sample_time, num_seconds, n_process):
    return er.EdfDatasetSegments(pre_cooldown=pre_cooldown, post_cooldown=post_cooldown, sample_time=sample_time, num_seconds=num_seconds, n_process=n_process)

@ex.capture
def valid_dataset_class(balance_valid_dataset):
    if balance_valid_dataset:
        return RULDataGenMultipleLabels
    else:
        return DataGenMultipleLabels

@ex.capture
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
    ex.add_artifact("/n/scratch2/ms994/seizure_multi_labels_edss_info.pkl")
    return train_edss, valid_edss, test_edss


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
def update_data(edss, seizure_classification_only, seizure_classes_to_use, include_seizure_type, include_montage_channels, zero_out_patients=False):
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

@ex.capture
def patient_func(tkn_file_paths, session_instead_patient):
    if session_instead_patient:
        return [read.parse_edf_token_path_structure(tkn_file_path)[1] + "/" + read.parse_edf_token_path_structure(tkn_file_path)[2] for tkn_file_path in tkn_file_paths]
    else:
        return [read.parse_edf_token_path_structure(tkn_file_path)[1] for tkn_file_path in tkn_file_paths]
@ex.capture
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
        train_edss, seizureLabels, patientInd, seizureSubtypes = reorder_channels(train_edss)
        valid_edss, validSeizureLabels, validPatientInd, validSeizureSubtypes = reorder_channels(valid_edss)
        test_edss, _, _, testSeizureSubtypes = reorder_channels(test_edss)
    elif include_seizure_type and include_montage_channels:
        train_edss, seizureLabels, patientInd, seizureSubtypes, montageLabels = reorder_channels(train_edss)
        valid_edss, validSeizureLabels, validPatientInd, validSeizureSubtypes, validMontageLabels = reorder_channels(valid_edss)
        test_edss, testSeizureLabels, testPatientInd, testSeizureSubtypes, testMontageLabels = reorder_channels(test_edss)
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

@ex.capture
def false_alarms_per_hour(fp, total_samps, num_seconds):
    num_chances_per_hour = 60 * 60 / num_seconds
    return (fp / total_samps) * num_chances_per_hour
