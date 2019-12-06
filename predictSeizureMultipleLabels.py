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
# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels, RULEdfDataGenerator, RULDataGenMultipleLabels
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like_pre_layers, conv2d_gridsearch_pre_layers
from keras import optimizers
from keras.layers import Dense, TimeDistributed, Input, Reshape, Dropout, LSTM, Flatten, Concatenate
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
ex = sacred.Experiment(name="seizure_conv_exp_domain_adapt_v2")

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

@ex.named_config
def knn():
    train_pkl = "/home/msaqib/train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/home/msaqib/valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/home/msaqib/test_multiple_labels_sessions_seizure_data_4.pkl"
    include_seizure_type = True
    session_instead_patient = True
    # max_bckg_samps_per_file = None
    # max_bckg_samps_per_file_test = None

@ex.named_config
def debug():
    train_pkl = "/home/ms994/debug_train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/home/ms994/debug_valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/home/ms994/debug_test_multiple_labels_seizure_data_4.pkl"
    max_bckg_samps_per_file = 5 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = 5
    max_samples=10000
    include_seizure_type=True
    session_instead_patient = True

@ex.named_config
def use_session_dbmi():
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_sessions_seizure_data_4.pkl"
    session_instead_patient = True
    include_seizure_type = True

@ex.named_config
def gnsz_fnsz():
    seizure_classes_to_use=["bckg", "gnsz", "fnsz"]

@ex.named_config
def gnsz():
    seizure_classes_to_use=["bckg", "gnsz"]

@ex.named_config
def fnsz():
    seizure_classes_to_use=["bckg", "fnsz"]

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


    precache = True
    regenerate_data = False
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_seizure_data_4.pkl"
    batch_size = 32

    # seizure_type = None

    pre_layer_h = 128
    num_lin_layer = 1

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
    include_seizure_type = False
    attach_seizure_type_to_seizure_detect = False
    seizure_classification_only = False
    seizure_classes_to_use = None
    update_seizure_detect_class_weights = False

    patient_weight = -1
    seizure_weight = 1

    num_post_lin_h = 5

    use_batch_normalization = True

    max_bckg_samps_per_file = 50 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = None
    max_samples=None
    use_standard_scaler = False
    use_filtering = True
    ref = "01_tcp_ar"
    combined_split = None
    lr = 0.005

    use_lstm = False
    use_time_layers_first = False
    max_pool_size_time = None

    epochs=100
    seizure_weight_decay = None

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
def get_data(mode, max_samples, n_process, max_bckg_samps_per_file, num_seconds, max_bckg_samps_per_file_test, include_seizure_type, ref="01_tcp_ar", num_files=None):
    if max_bckg_samps_per_file_test is None:
        max_bckg_samps_per_file_test = max_bckg_samps_per_file
    eds = getDataSampleGenerator()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()

    #increased n_process to deal with io processing
    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type)
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type)
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file_test, n_process=int(n_process*1.2), gap=num_seconds*pd.Timedelta(seconds=1), include_seizure_type=include_seizure_type)
    return train_edss, valid_edss, test_edss

@ex.capture
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
    attach_seizure_type_to_seizure_detect):
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
    if not use_lstm:
        y = Flatten()(y)

    for i in range(num_post_cnn_layers):
        y = Dense(num_post_lin_h, activation='relu')(y)
        y = Dropout(linear_dropout)(y)
    if not use_lstm:
        y_seizure_subtype = Dense(len(constants.SEIZURE_SUBTYPES), activation="softmax", name="seizure_subtype")(y)
        if include_seizure_type and attach_seizure_type_to_seizure_detect:
            y = Concatenate()([y, y_seizure_subtype])
        y_seizure = Dense(2, activation="softmax", name="seizure")(y)
        y_patient = Dense(num_patients, activation="softmax", name="patient")(y)
    else:
        y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)
        y_seizure = LSTM(2)(y)
        y_seizure_subtype = LSTM(len(constants.SEIZURE_SUBTYPES))(y)
        if include_seizure_type and attach_seizure_type_to_seizure_detect:
            raise Exception("Not IMPLEMENTED")
            y = Concatenate()([y, y_seizure_subtype])
        y_patient = LSTM(num_patients)(y)


    seizure_model = Model(inputs=x, outputs=[y_seizure])

    if include_seizure_type:
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

    seizure_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy"], metrics=["binary_accuracy"])
    if include_seizure_type:
        seizure_patient_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"], loss_weights=[seizure_weight,patient_weight, seizure_weight], metrics=["categorical_accuracy"])
    else:
        seizure_patient_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy",  "categorical_crossentropy"], loss_weights=[seizure_weight,patient_weight,], metrics=["categorical_accuracy"])

    patient_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
    return seizure_model, seizure_patient_model, patient_model, val_train_model

global_model = None

@ex.capture
def recompile_model(seizure_patient_model, epoch_num, seizure_weight, patient_weight, include_seizure_type, lr, seizure_weight_decay):
    if seizure_weight_decay is not None and (epoch_num + 1) % 1 == 0:
        weight_decay = seizure_weight_decay ** ((epoch_num + 1) / 1)
        print("Weight Decay! new Seizure Weight: {}".format(seizure_weight * weight_decay))

        if include_seizure_type:
            seizure_patient_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"], loss_weights=[seizure_weight * weight_decay,patient_weight, seizure_weight * weight_decay], metrics=["categorical_accuracy"])
        else:
            seizure_patient_model.compile(optimizers.Adam(lr=lr), loss=["categorical_crossentropy",  "categorical_crossentropy"], loss_weights=[seizure_weight * weight_decay, patient_weight,], metrics=["categorical_accuracy"])
    return seizure_patient_model



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
def update_data(edss, seizure_classification_only, seizure_classes_to_use):
    '''
    some of the tasks require different datasets and some filtering of the data i.e. only seizure classification or just some of the labels
    '''
    data = [datum[0] for datum in edss]
    patient_labels = [datum[1][1] for datum in edss]
    seizure_detection_labels = [datum[1][0] for datum in edss]
    seizure_class_labels = [datum[1][2] for datum in edss]
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
    for i, should_keep in enumerate(keep_index):
        if should_keep:
            data_to_keep.append(data[i])
            patient_labels_to_keep.append(patient_labels[i])
            seizure_detect_to_keep.append(seizure_detection_labels[i])
            seizure_class_labels_to_keep.append(seizure_class_labels[i])
    return [(data_to_keep[i], (seizure_detect_to_keep[i], patient_labels_to_keep[i], seizure_class_labels_to_keep[i])) for i in range(len(data_to_keep))], seizure_detect_to_keep, patient_labels_to_keep, seizure_class_labels_to_keep


@ex.capture
def get_data_generators(train_pkl,  valid_pkl, test_pkl, regenerate_data, use_standard_scaler, precache, batch_size, n_process, include_seizure_type, session_instead_patient,):
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

        patientInd = [datum[1][1] for datum in train_edss]
        seizureLabels = [datum[1][0] for datum in train_edss]
        if include_seizure_type:
            seizureSubtypes = [datum[1][2] for datum in train_edss]
            validSeizureSubtypes = [datum[1][2] for datum in valid_edss]
        validPatientInd = [datum[1][1] for datum in valid_edss]
        validSeizureLabels = [datum[1][0] for datum in valid_edss]
        allPatients = list(set([datum[1][1] for datum in train_edss]))
    else:
        print("(Re)generating data")
        train_edss, valid_edss, test_edss = get_data()
        tkn_file_paths = [train_edss.sampleInfo[key].token_file_path for key in train_edss.sampleInfo.keys()]
        if session_instead_patient:
            patients = [read.parse_edf_token_path_structure(tkn_file_path)[1] + "/" + read.parse_edf_token_path_structure(tkn_file_path)[2] for tkn_file_path in tkn_file_paths]
        else:
            patients = [read.parse_edf_token_path_structure(tkn_file_path)[1] for tkn_file_path in tkn_file_paths]
        allPatients = list(set(patients))
        patientInd = [allPatients.index(patient) for patient in patients]
        seizureLabels = [train_edss.sampleInfo[key].label for key in train_edss.sampleInfo.keys()]
        validSeizureLabels = [valid_edss.sampleInfo[key].label for key in valid_edss.sampleInfo.keys()]
        testSeizureLabels = [test_edss.sampleInfo[key].label for key in test_edss.sampleInfo.keys()]

        validPatientInd = [0 for i in range(len(validSeizureLabels))]
        if not include_seizure_type:
            for i in range(len(seizureLabels)):
                train_edss.sampleInfo[i].label = (seizureLabels[i], patientInd[i])
            for i in range(len(validSeizureLabels)):
                valid_edss.sampleInfo[i].label = (validSeizureLabels[i], 0) #the network has too many parameters if you include validation set patients (mutually exclusive) and the neural network should never choose validation patients anyways
        else:
            for i in range(len(seizureLabels)):
                train_edss.sampleInfo[i].label = (seizureLabels[i][0], patientInd[i], constants.SEIZURE_SUBTYPES.index(seizureLabels[i][1].lower()))
            for i in range(len(validSeizureLabels)):
                valid_edss.sampleInfo[i].label = (validSeizureLabels[i][0], 0, constants.SEIZURE_SUBTYPES.index(validSeizureLabels[i][1].lower())) #the network has too many parameters if you include validation set patients (mutually exclusive) and the neural network should never choose validation patients anyways
            for i in range(len(testSeizureLabels)):
                test_edss.sampleInfo[i].label = (testSeizureLabels[i][0], 0, constants.SEIZURE_SUBTYPES.index(testSeizureLabels[i][1].lower())) #the network has too many parameters if you include test set patients (mutually exclusive) and the neural network should never choose test patients anyways

        train_edss = train_edss[:]
        valid_edss = valid_edss[:]
        test_edss = test_edss[:]

        pkl.dump(train_edss[:], open(train_pkl, 'wb'))
        pkl.dump(valid_edss[:], open(valid_pkl, 'wb'))
        pkl.dump(test_edss[:], open(test_pkl, 'wb'))


    train_edss = update_data(train_edss)
    valid_edss = update_data(valid_edss)
    test_edss = update_data(test_edss)

    if use_standard_scaler:
        train_edss = read.EdfStandardScaler(
            train_edss, dataset_includes_label=True, n_process=n_process)[:]
        valid_edss = read.EdfStandardScaler(
            valid_edss, dataset_includes_label=True, n_process=n_process)[:]
        test_edss = read.EdfStandardScaler(
            test_edss, dataset_includes_label=True, n_process=n_process)[:]

    train_edss, seizureLabels, patientInd, seizureSubtypes = reorder_channels(train_edss)
    valid_edss, validSeizureLabels, validPatientInd, validSeizureSubtypes = reorder_channels(valid_edss)
    test_edss, _, _, testSeizureSubtypes = reorder_channels(test_edss)

    if include_seizure_type:
        edg = RULDataGenMultipleLabels(train_edss, num_labels=3, precache=True, batch_size=batch_size, labels=[seizureLabels, patientInd, seizureSubtypes], n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)),)
        valid_edg = DataGenMultipleLabels(valid_edss, num_labels=3, precache=True, batch_size=batch_size*4, labels=[validSeizureLabels, validPatientInd, validSeizureSubtypes], xy_tuple_form=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), shuffle=False)
        test_edg = DataGenMultipleLabels(test_edss, num_labels=3, precache=True, n_classes=(2, len(allPatients), len(constants.SEIZURE_SUBTYPES)), batch_size=batch_size*4, shuffle=False)
    else:
        edg = RULDataGenMultipleLabels(train_edss, num_labels=2, precache=True, labels=[seizureLabels, patientInd], batch_size=batch_size, n_classes=(2, len(allPatients)),) #learning means we are more likely to be affected by batch size, both for OOM in gpu and as a hyperparamer
        valid_edg = DataGenMultipleLabels(valid_edss, num_labels=2, precache=True, labels=[validSeizureLabels, validPatientInd], batch_size=batch_size*4, xy_tuple_form=True, n_classes=(2, len(allPatients)), shuffle=False) #batch size doesn't matter as much when we aren't learning but we still need batches to avoid OOM
        if len(test_edss[0][1]) > 1: #we throw out the seizure type label
            data = [datum[0] for datum in test_edss]
            labels = [datum[1][0] for datum in test_edss]
            test_edss = [(data[i], labels[i]) for i in range(len(data))]
        test_edg = EdfDataGenerator(test_edss, n_classes=2, precache=True, batch_size=batch_size, shuffle=False)
    return edg, valid_edg, test_edg, len(allPatients)

@ex.capture
def false_alarms_per_hour(fp, total_samps, num_seconds):
    num_chances_per_hour = 60 * 60 / num_seconds
    return (fp / total_samps) * num_chances_per_hour

@ex.main
def main(model_name, mode, num_seconds, imbalanced_resampler,  regenerate_data, epochs, fit_generator_verbosity, batch_size, n_process, steps_per_epoch, patience, include_seizure_type, max_bckg_samps_per_file_test, seizure_weight_decay, seizure_classification_only):
    class_weights = {0:1,1:1}
    edg, valid_edg, test_edg, len_all_patients = get_data_generators()

    print("Creating models")
    seizure_model, seizure_patient_model, patient_model, val_train_model = get_model(num_patients=len_all_patients)

    if regenerate_data:
        return

    # if steps_per_epoch is None:
    #     history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs)
    # else:
    #     history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs, steps_per_epoch=steps_per_epoch)


    # train_ordered_enqueuer = OrderedEnqueuer(edg, True)
    # valid_ordered_enqueuer = OrderedEnqueuer(valid_edg, True)


    num_epochs = epochs
    training_seizure_accs = []
    valid_seizure_accs = []
    train_patient_accs = []
    training_seizure_loss = []
    valid_seizure_loss = []

    oldPatientWeights = patient_model.layers[-1].get_weights()
    oldNonPatientWeights = [layer.get_weights() for layer in seizure_model.layers[:-1]]
    best_model_loss = -100
    patience_left = patience
    if include_seizure_type:
        subtype_accs = []
        subtype_losses = []
        valid_seizure_subtype_accs = []
        valid_seizure_subtype_loss = []

    for i in range(num_epochs):
        if patience_left == 0:
            continue
        recompile_model(seizure_patient_model, i)

        valid_labels_full_epoch = []
        valid_labels_epoch= []
        valid_predictions_full = []
        valid_predictions = []

        if include_seizure_type:
            subtype_epochs_accs = []
            subtype_val_epoch_labels = []
            subtype_val_predictions_epoch = []
            subtype_val_epoch_labels_full = []
            subtype_val_predictions_epoch_full = []





        train_seizure_loss_epoch = []
        train_subtype_loss_epoch = []

        seizure_accs = []
        patient_accs_epoch = []
        # for j in range(len(edg)):
        if steps_per_epoch is None:
            steps_per_epoch = len(edg)
        for j in range(steps_per_epoch):

            train_batch = edg[j]
            data_x = train_batch[0]
            data_x = data_x.astype(np.float32)
            data_x = np.nan_to_num(data_x)
            if include_seizure_type:
                loss, seizure_loss, patient_loss, subtype_loss, seizure_acc, patient_acc, subtype_acc = seizure_patient_model.train_on_batch(data_x, train_batch[1])
                subtype_epochs_accs.append(subtype_acc)

            else:
                loss, seizure_loss, patient_loss, seizure_acc, patient_acc = seizure_patient_model.train_on_batch(data_x, train_batch[1])
            seizure_accs.append(seizure_acc)
            #old patient weights are trying to predict for patient, try to do the prediction!
            patient_model.layers[-1].set_weights(oldPatientWeights)
            #keep the other nonpatient weights which try not to predict for patient!
            oldNonPatientWeights = [layer.get_weights() for layer in seizure_model.layers[:-1]]
            patient_loss, patient_acc = patient_model.train_on_batch(train_batch[0], train_batch[1][1])
            patient_accs_epoch.append(patient_acc)

            train_seizure_loss_epoch.append(seizure_loss)
            if include_seizure_type:
                train_subtype_loss_epoch.append(subtype_loss)

            #get weights that try to predict for patient
            oldPatientWeights = patient_model.layers[-1].get_weights()

            #set weights that don't ruin seizure prediction
            for layer_num, layer in enumerate(seizure_model.layers[:-1]):
                seizure_model.layers[layer_num].set_weights(oldNonPatientWeights[layer_num])
            if (j % 100) == 0:
                printEpochUpdateString = "epoch: {} batch: {}/{}, seizure acc: {}, patient acc: {}, loss: {}".format(i, j, len(edg), np.mean(seizure_accs), np.mean(patient_accs_epoch), loss)
                if include_seizure_type:
                    printEpochUpdateString += ", seizure subtype acc: {}, subtype loss: {}".format(np.mean(subtype_epochs_accs), np.mean(train_subtype_loss_epoch))
                print(printEpochUpdateString)
    #     valid_edg.start_background()

        assert valid_labels_epoch == []
        assert valid_predictions == []

        for j in range(len(valid_edg)):
            valid_batch = valid_edg[j]
            data_x = valid_batch[0]
            data_x = data_x.astype(np.float32)
            data_x = np.nan_to_num(data_x) #ssome weird issue with incorrect data conversion


            val_batch_predictions = val_train_model.predict_on_batch(data_x)
            if include_seizure_type:
                subtype_val_predictions_epoch.append(val_batch_predictions[1].argmax(1))
                subtype_val_predictions_epoch_full.append(val_batch_predictions[1])
                subtype_val_epoch_labels.append(valid_batch[1][2].argmax(1))
                subtype_val_epoch_labels_full.append(valid_batch[1][2])
                valid_labels_epoch.append(valid_batch[1][0].argmax(1))
                valid_labels_full_epoch.append(valid_batch[1][0])
                valid_predictions.append(val_batch_predictions[0].argmax(1))
                valid_predictions_full.append(val_batch_predictions[0])
            else:
                valid_labels_epoch.append(valid_batch[1][0].argmax(1))
                valid_labels_full_epoch.append(valid_batch[1][0])
                valid_predictions.append(val_batch_predictions.argmax(1))
                valid_predictions_full.append(val_batch_predictions)

        def get_sum_seizures():
            num_seizures = 0
            for j in range(len(valid_edg)):
                valid_batch = valid_edg[j]
                num_seizures += valid_batch[1][0].argmax(1).sum()
            return num_seizures

        #random infinitye predictions? I'm assuming some weird type conversion issues and that nan_to_num should fix this

        valid_labels_epoch= np.nan_to_num(np.hstack(valid_labels_epoch).astype(np.float32))
        valid_predictions = np.nan_to_num(np.hstack(valid_predictions).astype(np.float32))

        print("debug: valid_labels_epoch shape {}, valid_predictions.shape {}".format(valid_labels_epoch.shape, valid_predictions.shape))
        print("We predicted {} seizures in the validation split, there were actually {}".format(valid_predictions.sum(), valid_labels_epoch.sum()))
        print("We predicted {} seizure/total in the validation split, there were actually {}".format(valid_predictions.sum()/len(valid_predictions), valid_labels_epoch.sum()/len(valid_labels_epoch)))
        print(classification_report(valid_labels_epoch, valid_predictions))




        valid_labels_full_epoch = np.nan_to_num(np.vstack(valid_labels_full_epoch).astype(np.float32))
        valid_predictions_full = np.nan_to_num(np.vstack(valid_predictions_full).astype(np.float32))

        if include_seizure_type:
            subtype_val_epoch_labels = np.nan_to_num(np.hstack(subtype_val_epoch_labels).astype(np.float32))
            subtype_val_predictions_epoch = np.nan_to_num(np.hstack(subtype_val_predictions_epoch).astype(np.float32))
            subtype_val_epoch_labels_full = np.nan_to_num(np.vstack(subtype_val_epoch_labels_full).astype(np.float32))
            subtype_val_predictions_epoch_full = np.nan_to_num(np.vstack(subtype_val_predictions_epoch_full).astype(np.float32))



        try:
            auc = roc_auc_score(valid_predictions, valid_labels_epoch)
        except Exception:
            auc = "undefined"
        valid_acc =  accuracy_score(valid_predictions, valid_labels_epoch)
        valid_seizure_accs.append(valid_acc)
        train_patient_accs.append(np.mean(patient_accs_epoch))
        valid_loss = log_loss(valid_labels_full_epoch, valid_predictions_full)
        training_seizure_loss.append(np.mean(train_seizure_loss_epoch))
        printEpochEndString = "end epoch: {}, f1: {}, auc: {}, acc: {}, loss: {}\n".format(i, f1_score(valid_predictions, valid_labels_epoch), auc, valid_acc, valid_loss)
        valid_seizure_loss.append(valid_loss)
        if include_seizure_type:
            subtype_losses.append(np.mean(train_subtype_loss_epoch))
            subtype_acc = np.mean(subtype_epochs_accs)
            subtype_accs.append(subtype_acc)
            val_subtype_acc = accuracy_score(subtype_val_epoch_labels, subtype_val_predictions_epoch)
            valid_seizure_subtype_accs.append(val_subtype_acc)
            val_subtype_loss = log_loss(subtype_val_epoch_labels_full, subtype_val_predictions_epoch_full)
            valid_seizure_subtype_loss.append(val_subtype_loss)
            macro_subtype_f1 = f1_score(subtype_val_epoch_labels, subtype_val_predictions_epoch, average='macro')
            weighted_subtype_f1 = f1_score(subtype_val_epoch_labels, subtype_val_predictions_epoch, average='weighted')


            printEpochEndString += "\tsubtype info: train acc: {}, valid acc:{}, loss: {}, macro_f1: {}, weighted_f1: {}".format(subtype_acc, val_subtype_acc, val_subtype_loss, macro_subtype_f1, weighted_subtype_f1)



        print(printEpochEndString)
        if seizure_classification_only:
            new_val_f1 = weighted_subtype_f1
        else:
            new_val_f1 = f1_score(valid_predictions, valid_labels_epoch)
        if (new_val_f1 > best_model_loss):
            patience_left = patience
            best_model_loss = new_val_f1
            try:
                val_train_model.save(model_name)
                print("improved val score to {}".format(best_model_loss))
            except Exception as e:
                print("{}\n".format(e))
                print("failed saving\n")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early Stopping!")




        training_seizure_accs.append(np.mean(seizure_accs))

        edg.on_epoch_end()
        valid_edg.on_epoch_end()

    model = load_model(model_name)

    y_pred = model.predict_generator(test_edg)


    results = Dict()
    results.history = Dict({
        "binary_accuracy": training_seizure_accs,
        "val_binary_accuracy": valid_seizure_accs,
        "seizure_loss": training_seizure_loss,
        "valid_seizure_loss": valid_seizure_loss,
        "patient_acc": train_patient_accs,

    })
    if include_seizure_type:
        results.history.subtype.acc = subtype_accs
        results.history.subtype.val_acc = valid_seizure_subtype_accs
        results.history.subtype.loss = subtype_losses
        results.history.subtype.val_loss = valid_seizure_subtype_loss

    if include_seizure_type:
        y_seizure_label =  np.array([data[1][0] for data in test_edg.dataset]).astype(int)
        y_seizure_pred = np.array([y_pred[0].argmax(1)]).astype(int)[0]
        y_subtype_label =  np.array([data[1][2] for data in test_edg.dataset]).astype(int)
        y_subtype_pred = np.array([y_pred[1].argmax(1)]).astype(int)[0]
        results.subtype.acc = accuracy_score(y_subtype_label, y_subtype_pred)
        results.subtype.f1.macro = f1_score(y_subtype_label, y_subtype_pred, average='macro')
        results.subtype.f1.micro = f1_score(y_subtype_label, y_subtype_pred, average='micro')
        results.subtype.f1.weighted = f1_score(y_subtype_label, y_subtype_pred, average='weighted')
        results.subtype.confusion_matrix = confusion_matrix(y_subtype_pred, y_subtype_label)
        results.subtype.classification_report = classification_report(y_subtype_pred, y_subtype_label, output_dict=True)
    else:
        y_seizure_label =  np.array([data[1] for data in test_edg.dataset]).astype(int)
        y_seizure_pred = np.array(y_pred.argmax(1)).astype(int)

    print("We predicted {} seizures in the test split, there were actually {}".format(y_seizure_pred.sum(), np.array([data[1] for data in test_edg.dataset]).astype(int).sum()))
    print("We predicted {} seizure/total in the test split, there were actually {}".format(y_seizure_pred.sum()/len(y_seizure_pred), np.array([data[1] for data in test_edg.dataset]).astype(int).sum()/len(np.array([data[1] for data in test_edg.dataset]).astype(int))))

    if not seizure_classification_only:
        results.seizure.acc = accuracy_score(y_seizure_pred, y_seizure_label)
        results.seizure.f1 = f1_score(y_seizure_pred, y_seizure_label)
        results.seizure.classification_report = classification_report(y_seizure_label, y_seizure_pred, output_dict=True)
        results.seizure.confusion_matrix = confusion_matrix(y_seizure_label, y_seizure_pred)
        if max_bckg_samps_per_file_test is not None:
            total_samps = sum(results.seizure.confusion_matrix[0]) #just use the samps labeled negative, max_bckg_samps_per_file_test is used to run faster but leads to issues with class imbalance not being fully reflected if we include seizure
        else:
            total_samps = sum(sum(results.seizure.confusion_matrix))
        results.seizure.false_alarms_per_hour = false_alarms_per_hour(results.seizure.confusion_matrix[0][1], total_samps=total_samps)

        try:
            results.seizure.AUC = roc_auc_score(y_seizure_pred, y_seizure_label)
        except Exception:
            results.seizure.AUC = "failed to calculate"

    return results.to_dict()


if __name__ == "__main__":
    ex.run_commandline()
