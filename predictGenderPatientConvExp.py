from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
from os import path
import data_reader as read
import constants
import util_funcs
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like_pre_layers
from keras import optimizers
from keras.layers import Dense, TimeDistributed, Input, Reshape
from keras.models import Model
from keras.utils import multi_gpu_model
import pickle as pkl
import sacred
# from sacred.stflow import LogFileWriter
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
ex = sacred.Experiment(name="gender_patient_predict_conv_gridsearch")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))



allPatients = None


@ex.named_config
def conv_spatial_filter_2_2():
    conv_spatial_filter = (2, 2)


@ex.named_config
def conv_spatial_filter_3_3():
    conv_spatial_filter = (3, 3)

@ex.named_config
def max_pool_size_2_2():
    max_pool_size = (2,2)

@ex.named_config
def max_pool_size_1_2():
    max_pool_size = (1,2)


@ex.named_config
def conv_temporal_filter_1_7():
    conv_temporal_filter = (1, 7)

@ex.named_config
def conv_temporal_filter_1_3():
    conv_temporal_filter = (1, 3)


@ex.named_config
def conv_temporal_filter_2_7():
    conv_temporal_filter = (2,7)

@ex.named_config
def conv_temporal_filter_2_5():
    conv_temporal_filter = (2,5)

@ex.named_config
def conv_temporal_filter_2_3():
    conv_temporal_filter = (2, 3)


@ex.named_config
def debug():
    num_files = 200
    batch_size = 16
    num_epochs = 20
    max_num_samples = 2


@ex.named_config
def standardized_combined_split_patient():
    use_combined = True
    use_random_ensemble = True
    precached_train_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_train_data_patient_gender_patient_split.pkl"
    precached_valid_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_valid_data_patient_gender_patient_split.pkl"
    precached_test_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_test_data_patient_gender_patient_split.pkl"
    ensemble_sample_info_path = "/n/scratch2/ms994/standardized_edf_ensemble_sample_info_patient_gender_patient_split.pkl"
    patient_path = "/n/scratch2/ms994/patient_list_with_gender_2_plus_sess_patient.pkl"
    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True
    split_on_session = False

@ex.named_config
def standardized_combined_simple_ensemble():
    use_combined = True
    use_random_ensemble = True
    precached_train_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_train_data_patient_gender.pkl"
    precached_valid_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_valid_data_patient_gender.pkl"
    precached_test_pkl = "/n/scratch2/ms994/standardized_combined_simple_ensemble_test_data_patient_gender.pkl"
    ensemble_sample_info_path = "/n/scratch2/ms994/standardized_edf_ensemble_sample_info_patient_gender.pkl"
    patient_path = "/n/scratch2/ms994/patient_list_with_gender_2_plus_sess.pkl"
    max_num_samples = 5  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True
    split_on_session = True


@ex.named_config
def stop_on_training_loss():
    early_stopping_on = "loss"

@ex.named_config
def use_lin_pre():
    use_time_first = True
    max_pool_size = (2, 1)
    max_pool_stride = (2, 1)
    use_linear_pre_layers = True
    num_lin_layer = 1
    pre_layer_h = 32
    num_layers = 4

@ex.named_config
def use_lin_pre_max_pool():
    use_time_first = True
    max_pool_size = (3, 2)
    max_pool_stride = (3, 2)
    use_linear_pre_layers = True
    num_lin_layer = 2
    pre_layer_h = 128
    num_layers = 3


@ex.config
def config():
    n_process = 8
    num_files = None
    max_length = 4 * constants.COMMON_FREQ
    batch_size = 32
    start_offset_seconds = 0  # matters if we aren't doing random ensemble sampling
    dropout = 0.25
    use_early_stopping = True
    patience = 10
    patient_path = "/n/scratch2/ms994/patient_list_with_gender_2_plus_sess.pkl"
    model_name = "/n/scratch2/ms994/out/" + randomString() + ".h5" #set to rando string so we don't have to worry about collisions
    precached_train_pkl = "train_data.pkl"
    precached_valid_pkl = "valid_data.pkl"
    precached_test_pkl = "test_data.pkl"
    num_epochs = 1000
    lr = 0.0002
    validation_size = 0.2
    test_size = 0.2
    use_cached_pkl = True
    use_vp = True
    # for custom architectures
    num_layers = 4
    num_filters = 20
    num_temporal_filter = 1
    use_filtering = True
    max_pool_size = (1, 2)
    max_pool_stride = (1, 2)
    patient_weight=1
    gender_weight=1
    use_time_first = False
    use_batch_normalization = True
    use_random_ensemble = False
    max_num_samples = 10  # number of samples of eeg data segments per eeg.edf file
    use_combined = False
    combined_split = "combined"
    num_gpus = 1
    early_stopping_on = "val_loss"
    test_train_split_pkl_path = "train_test_split_info.pkl"
    # if use_cached_pkl is false and this is true, just generates pickle files, doesn't make models or anything
    regenerate_data = False
    use_standard_scaler = False
    ensemble_sample_info_path = "edf_ensemble_path.pkl"
    fit_generator_verbosity = 2
    steps_per_epoch = None
    validation_steps = None
    shuffle_generator = True
    use_dl = True
    use_inception_like=False
    shuffle_channels=False
    use_linear_pre_layers = False
    pre_layer_h = 64
    num_lin_layer = None
    split_on_session = True


# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))


@ex.capture
def get_model_checkpoint(model_name, monitor='val_loss'):
    return ModelCheckpoint(model_name, monitor=monitor, save_best_only=True, verbose=1)


@ex.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on)

@ex.capture
def get_cb_list():
    return [get_model_checkpoint(), get_early_stopping()]

@ex.capture
def get_model(use_linear_pre_layers, pre_layer_h, use_time_first, num_lin_layer, num_filters, dropout, num_layers, num_gpus, patient_weight, gender_weight, max_pool_size, max_pool_stride, lr=0.0005):
    if not use_linear_pre_layers:
        return get_no_linear_pre_model()
    else:
        assert use_time_first
        #make a linear pre layer to learn a linear combination of channels that represents the optimal input
        x = Input((500, 21, 1)) #time, ecg channel, cnn channel
        y = Reshape((500, 21))(x) #remove channel dim
        y = TimeDistributed(Dense(pre_layer_h, activation="relu"))(y)
        for i in range(num_lin_layer - 1):
            y = TimeDistributed(Dense(pre_layer_h, activation="relu"))(y)
        y = Reshape((500, pre_layer_h, 1))(y) #add back in channel dim


        _, y = inception_like_pre_layers(x=y, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride, num_filters=num_filters, dropout=dropout, num_layers=num_layers)
        y_gender = Dense(2, activation="softmax", name="gender")(y)
        y_patient = Dense(len(allPatients), activation="softmax", name="patient")(y)
        model = Model(inputs=x, outputs=[y_gender, y_patient])
        model.summary()
        if num_gpus is not None and num_gpus > 1:
            model = multi_gpu_model(model, num_gpus)
        adam = keras.optimizers.Adam(lr=lr)
        loss_weights = [np.float(gender_weight), np.float(patient_weight)]
        model.compile(adam, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=["categorical_accuracy"], loss_weights=loss_weights)
        return model


@ex.capture
def get_no_linear_pre_model(num_filters, dropout, num_layers, num_gpus, patient_weight, gender_weight, max_pool_size, max_pool_stride, lr=0.0005):
    '''
    Created before realizing that I'd want to try to make a FC layers beforehand applied to each time step
    '''
    x, y = inception_like_pre_layers(input_shape=(21, 500, 1), max_pool_size=max_pool_size, max_pool_stride=max_pool_stride, num_filters=num_filters, dropout=dropout, num_layers=num_layers)
    y_gender = Dense(2, activation="softmax", name="gender")(y)
    y_patient = Dense(len(allPatients), activation="softmax", name="patient")(y)
    model = Model(inputs=x, outputs=[y_gender, y_patient])
    model.summary()
    if num_gpus is not None and num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)
    adam = keras.optimizers.Adam(lr=lr)
    loss_weights = [np.float(gender_weight), np.float(patient_weight)]
    model.compile(adam, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=["categorical_accuracy"], loss_weights=loss_weights)
    return model

@ex.capture
def get_data_generator(precached_train_pkl, precached_valid_pkl, precached_test_pkl, patient_path, batch_size, use_cached_pkl, shuffle_channels, use_time_first):
    global allPatients
    if path.exists(precached_test_pkl) \
        and path.exists(precached_train_pkl) \
        and path.exists(precached_valid_pkl) \
        and use_cached_pkl:
            allPatients = pkl.load(open(patient_path, 'rb'))
            trainEnsemblerRawData = pkl.load(open(precached_train_pkl, 'rb'))
            testEnsemblerRawData = pkl.load(open(precached_test_pkl, 'rb'))
            validEnsemblerRawData = pkl.load(open(precached_valid_pkl, 'rb'))
    else:
        trainEnsembler, trainEnsemblerRawData, testEnsembler, testEnsemblerRawData, validEnsemblerRawData = get_raw_data()
        pkl.dump(trainEnsemblerRawData, open(precached_train_pkl, 'wb'))
        pkl.dump(validEnsemblerRawData, open(precached_valid_pkl, 'wb'))
        pkl.dump(testEnsemblerRawData, open(precached_test_pkl, 'wb'))
        pkl.dump(allPatients, open(patient_path, 'wb'))
    testDataGen = DataGenMultipleLabels(testEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=use_time_first, shuffle_channels=shuffle_channels)
    trainDataGen = DataGenMultipleLabels(trainEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=use_time_first, shuffle=True, shuffle_channels=shuffle_channels)
    validDataGen = DataGenMultipleLabels(validEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=use_time_first, shuffle_channels=shuffle_channels)
    return trainDataGen, validDataGen, testDataGen, testEnsemblerRawData

@ex.capture
def get_raw_data(num_files, max_num_samples, n_process, split_on_session, test_size, validation_size):
    '''
    reality is that multitask learning can't actually incorporate both gender and patient
    in a fair way; either patients need to be segregated between test and train sets, in which case
    the predict patient task is impossible on the test set, or the patients are shared and sessions are
    segregated, in which case the predict gender task becomes unfair
    if split_on_session is True, we prioritize patient id, otherwise, we don't
    '''
    global allPatients
    files, genders = cta.demux_to_tokens(cta.getGenderAndFileNames("combined", "01_tcp_ar", True))
    files = files[:num_files]
    genders = genders[:num_files]
    sessions = [read.parse_edf_token_path_structure(file)[2] for file in files]
    patients = [read.parse_edf_token_path_structure(file)[1] for file in files]
    allPatients = list(set(patients))
    allPatients.sort()
    sessionDict = Dict()
    for i, file in enumerate(files):
        sessionDict[patients[i]][sessions[i]][i].file = file
        sessionDict[patients[i]][sessions[i]][i].gender = genders[i]
        sessionDict[patients[i]][sessions[i]][i].patient = allPatients.index(patients[i])
    for patient in set(patients):
        #delete all patients with only one session if we are doing a patient id
        if split_on_session:
            if len(sessionDict[patient].keys()) < 2:
                del sessionDict[patient]


    def returnFilesAndLabelsFromSessionDict(d):
        files = []
        labels = []
        for id_num in d.keys():
            files.append(d[id_num].file)
            labels.append((d[id_num].gender, d[id_num].patient))
        return files, labels
    testPatientFiles = []
    testLabels = []
    trainPatientFiles = []
    trainLabels = []
    validPatientFiles = []
    validLabels = []

    if split_on_session:
        for patient in sessionDict.keys():
            testSessionToAdd = np.random.choice(list(sessionDict[patient].keys()))
            for session in sessionDict[patient].keys():
                files, labels = returnFilesAndLabelsFromSessionDict(sessionDict[patient][session])
                if session == testSessionToAdd:
                    testPatientFiles += files
                    testLabels += labels
                else:
                    trainPatientFiles += files
                    trainLabels += labels
    else:
        trainPatients, testPatients = train_test_split(list(sessionDict.keys()), test_size=test_size)
        trainPatients, validPatients = train_test_split(trainPatients, test_size=validation_size)

        for patient in sessionDict.keys():
            for session in sessionDict[patient].keys():
                files, labels = returnFilesAndLabelsFromSessionDict(sessionDict[patient][session])
                if patient in trainPatients:
                    trainPatientFiles += files
                    trainLabels += labels
                elif patient in validPatients:
                    validPatientFiles += files
                    validLabels += labels
                else:
                    testPatientFiles += files
                    testLabels += labels

    testEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=max_num_samples, edf_tokens=testPatientFiles, n_process=n_process, labels=testLabels, filter=True)
    testEnsemblerRawData = testEnsembler[:]

    trainEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=max_num_samples, edf_tokens=trainPatientFiles, n_process=n_process, labels=trainLabels, filter=True)
    trainEnsemblerRawData = trainEnsembler[:]

    if split_on_session:
        trainEnsemblerRawData, validEnsemblerRawData = train_test_split(trainEnsemblerRawData, test_size=0.1)
    else:
        validEnsemblerRawData = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=max_num_samples, edf_tokens=validPatientFiles, n_process=n_process, labels=validLabels, filter=True)

    return trainEnsembler, trainEnsemblerRawData, testEnsembler, testEnsemblerRawData, validEnsemblerRawData[:]

@ex.main
# @LogFileWriter(ex)
def main(regenerate_data, num_epochs, fit_generator_verbosity, model_name, num_gpus, use_time_first, steps_per_epoch):
    trainDataGen, validDataGen, testDataGen, testEnsemblerRawData = get_data_generator()
    if regenerate_data:
        return
    model = get_model()
    cb_list = get_cb_list()
    if steps_per_epoch is not None:
        history = model.fit_generator(trainDataGen, epochs=num_epochs, validation_data=validDataGen, callbacks=cb_list, steps_per_epoch=steps_per_epoch, verbose=fit_generator_verbosity)
    else:
        history = model.fit_generator(trainDataGen, epochs=num_epochs, validation_data=validDataGen, callbacks=cb_list, verbose=fit_generator_verbosity)

    results = Dict()
    results.history = history.history

    model = keras.models.load_model(model_name)
    if num_gpus is not None and num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)

    if use_time_first:
        transpose = (0, 1, 2)
    else:
        transpose = (1, 0, 2) #don't need to transpose actually

    y_pred = model.predict(np.stack([data[0].reshape(500,21,1).transpose(*transpose) for data in testEnsemblerRawData]))

    results.gender.acc = accuracy_score(y_pred[0].argmax(1), [data[1][0] for data in testEnsemblerRawData])
    results.gender.AUC = roc_auc_score(y_pred[0].argmax(1), [data[1][0] for data in testEnsemblerRawData])

    testPatients = [data[1][1] for data in testEnsemblerRawData]
    results.patient.acc = accuracy_score(y_pred[1].argmax(1), testPatients)

    def top_k_acc(k):
        top_k_pred = y_pred[1].argsort()[:,-(k):]
        return np.mean([testPatients[i] in top_k_pred[i] for i in range(len(testPatients))])

    results.patient.multi_k_acc = {
        "1":top_k_acc(1),
        "2":top_k_acc(2),
        "5":top_k_acc(5),
        "10":top_k_acc(10),
        "20":top_k_acc(20)
    }

    return results.to_dict()


if __name__ == "__main__":
    ex.run_commandline()
