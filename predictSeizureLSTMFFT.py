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

# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.config
def config():
    model_name = "/n/scratch2/ms994/out/" + randomString() + ".h5" #set to rando string so we don't have to worry about collisions
    mode=er.EdfDatasetSegmentedSampler.DETECT_MODE
    max_samples=None
    max_pool_size = (2,2)
    max_pool_stride = (2,2)
    steps_per_epoch = None
    session_instead_patient=False

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
    optimizer_name="adam"
    lstm_h = 128
    lstm_return_sequence = False
    reduce_lr_on_plateau = False
    change_batch_size_over_time = None
    add_gaussian_noise = None

    precache = True
    regenerate_data = False
    processed_train_pkl = "/n/scratch2/ms994/processed_train_multiple_labels_seizure_data_4.pkl"
    processed_valid_pkl = "/n/scratch2/ms994/processed_valid_multiple_labels_seizure_data_4.pkl"
    processed_test_pkl = "/n/scratch2/ms994/processed_test_multiple_labels_seizure_data_4.pkl"

    train_pkl = "/n/scratch2/ms994/train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_seizure_data_4.pkl"
    batch_size = 32

    # seizure_type = None

    pre_layer_h = 128
    num_lin_layer = 1

    patience=5
    early_stopping_on="val_loss"
    fit_generator_verbosity = 2
    num_layers = 3
    num_filters = 1
    num_temporal_filter=1
    num_post_cnn_layers = 2
    hyperopt_run = False
    make_model_in_parallel = False
    randomly_reorder_channels = False #use if we want to try to mess around with EEG order
    include_seizure_type = False
    attach_seizure_type_to_seizure_detect = False
    seizure_classification_only = False
    seizure_classes_to_use = None
    update_seizure_class_weights = False
    min_seizure_weight = 0
    model_type = None

    patient_weight = -1
    seizure_weight = 1
    complex_feature_channels=constants.SYMMETRIC_COLUMN_SUBSET


    num_post_lin_h = 5

    use_batch_normalization = True

    max_bckg_samps_per_file = 50 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = -1 #reflect the full imbalance in the dataset
    max_samples=None
    use_standard_scaler = False
    use_filtering = True
    ref = "01_tcp_ar"
    combined_split = None
    lr = 0.005
    lr_decay = 0

    use_lstm = False
    use_time_layers_first = False
    max_pool_size_time = None
    validation_f1_score_type = None



    balance_valid_dataset = False

    epochs=100
    seizure_weight_decay = None
    # measure_train_patient_bias = False

    test_patient_model_after_train = False
    train_patient_model_after_train = False
    valid_patient_model_after_train = False
    random_rearrange_each_batch = False
    random_rescale = False
    rescale_factor = 1.3
    include_montage_channels = False
    coherence_bin = pd.Timedelta(seconds=1)

    time_step = pd.Timedelta(seconds=0.5)

@ex.named_config
def debug():
    train_pkl = "/home/ms994/debug_train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/home/ms994/debug_valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/home/ms994/debug_test_multiple_labels_seizure_data_4.pkl"
    processed_train_pkl = "/n/scratch2/ms994/debug_processed_train_multiple_labels_seizure_data_4.pkl"
    processed_valid_pkl = "/n/scratch2/ms994/debug_processed_valid_multiple_labels_seizure_data_4.pkl"
    processed_test_pkl = "/n/scratch2/ms994/debug_processed_test_multiple_labels_seizure_data_4.pkl"
    max_bckg_samps_per_file = 5 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = 5
    max_samples=10000
    include_seizure_type=True
    session_instead_patient = True

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
def process_data(edss, time_step, n_process):
    coherData = wfdata.CoherenceTransformer(simple_edss(edss), is_pandas=False, is_tuple_data=True, average_coherence=False, coherence_bin=time_step, n_process=n_process)[:]
    fftData = read.EdfFFTDatasetTransformer(simple_edss(edss), is_tuple_data=True, is_pandas_data=False, freq_bins=constants.FREQ_BANDS, window_size=time_step, n_process=n_process)[:]
    fftData = [fftDatum[0].transpose((1, 0,2)).reshape(8, fftDatum[0].shape[0] *  fftDatum[0].shape[2]) for fftDatum in fftData]
    labels = [coherDatum[1] for coherDatum in coherData]
    coherData = [coherDatum[0] for coherDatum in coherData]
    data_x = [np.hstack([coherData[i], fftData[i]]) for i in range(len(fftData))]
    return list(zip(data_x, labels))


@ex.capture
def simple_edss(edss, complex_feature_channels):
    '''
    Use only a few columns so that we don't make 21*20 coherence pairs
    '''
    all_channels = util_funcs.get_common_channel_names()
    subset_channels = [all_channels.index(channel) for channel in complex_feature_channels]
    return [(datum[0][:, subset_channels], datum[1]) for datum in edss]



@ex.capture
def get_data_generators(train_pkl,  valid_pkl, test_pkl, processed_train_pkl, processed_valid_pkl, processed_test_pkl, \
                 regenerate_data, use_standard_scaler, precache, n_process, include_seizure_type, include_montage_channels, batch_size):
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
        test_edss, testSeizureLabels, testPatientInd, testSeizureSubtypes = (test_edss)
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
        if path.exists(processed_train_pkl) and precache:
            processed_train_data = pkl.load(open(processed_train_pkl, "rb"))
            processed_valid_data = pkl.load(open(processed_valid_pkl, "rb"))
            processed_test_data = pkl.load(open(processed_test_pkl, "rb"))
        else:
            processed_train_data = process_data(train_edss)
            processed_valid_data = process_data(valid_edss)
            processed_test_data = process_data(test_edss)
            pkl.dump(processed_train_data, open(processed_train_pkl, "wb"))
            pkl.dump(processed_valid_data, open(processed_valid_pkl, "wb"))
            pkl.dump(processed_test_data, open(processed_test_pkl, "wb"))
        edg = RULDataGenMultipleLabels(processed_train_data, num_labels=3, precache=True, batch_size=batch_size, labels=[seizureLabels, patientInd, seizureSubtypes], n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)),)
        valid_edg = valid_dataset_class()(processed_valid_data, num_labels=3, precache=True, batch_size=batch_size*4, labels=[validSeizureLabels, validPatientInd, validSeizureSubtypes], xy_tuple_form=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), shuffle=False)
        test_edg = DataGenMultipleLabels(processed_test_data, num_labels=3, precache=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), batch_size=batch_size*4, shuffle=False)

        return edg, valid_edg, test_edg, len(allPatients)
    elif include_seizure_type and include_montage_channels:
        raise Exception("Not Implemented")
    elif not include_seizure_type and not include_montage_channels:
        raise Exception("Not Implemented")
    raise Exception("Not Implemented")

    # return edg, valid_edg, test_edg, len(allPatients)

@ex.capture
def false_alarms_per_hour(fp, total_samps, num_seconds):
    num_chances_per_hour = 60 * 60 / num_seconds
    return (fp / total_samps) * num_chances_per_hour

@ex.main
def main(model_name):
    edg, valid_edg, test_edg, len_all_patients = get_data_generators()
    raise Exception()


if __name__ == "__main__":
    ex.run_commandline()
