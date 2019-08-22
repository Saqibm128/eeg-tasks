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
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like
from keras import optimizers
import pickle as pkl
import sacred
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
ex = sacred.Experiment(name="gender_predict_conv_gridsearch")

# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))



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


@ex.named_config
def stop_on_training_loss():
    early_stopping_on = "loss"


@ex.config
def config():
    n_process = 8
    num_files = None
    max_length = 4 * constants.COMMON_FREQ
    batch_size = 64
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
    num_filters = 100
    num_temporal_filter = 1
    use_filtering = True
    max_pool_size = (1, 2)
    max_pool_stride = (1, 2)
    patient_weight=1
    gender_weight=1
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
def get_model(num_filters, dropout, num_layers, num_gpus, patient_weight, gender_weight, lr=0.0005):
    x, y = cnn_models.inception_like_pre_layers(input_shape=(21, 500, 1), num_filters=num_filters, dropout=dropout, num_layers=num_layers)
    y_gender = Dense(2, activation="softmax", name="gender")(y)
    y_patient = Dense(len(allPatients), activation="softmax", name="patient")(y)
    model = Model(inputs=x, outputs=[y_gender, y_patient])
    model.summary()
    if num_gpus is not None or num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)
    adam = keras.optimizers.Adam(lr=lr)
    loss_weights = [gender_weight, patient_weight]
    model.compile(adam, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=["categorical_accuracy"], loss_weights=loss_weights)
    return model

@ex.capture
def get_data_generator(precached_train_pkl, precached_valid_pkl, precached_test_pkl, patient_path, batch_size, use_cached_pkl):
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
    testDataGen = DataGenMultipleLabels(testEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False)
    trainDataGen = DataGenMultipleLabels(trainEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False, shuffle=True)
    validDataGen = DataGenMultipleLabels(validEnsemblerRawData, batch_size=batch_size, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False)
    return trainDataGen, validDataGen, testDataGen, testEnsemblerRawData

@ex.capture
def get_raw_data(num_files, max_num_samples, n_process):
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
    #delete all patients with only one session
    for patient in set(patients):
        print(len(sessionDict[patient].keys()))
        if len(sessionDict[patient].keys()) < 2:
            del sessionDict[patient]


    def returnFilesAndLabelsFromSessionDict(d):
        files = []
        genders = []
        for id_num in d.keys():
            files.append(d[id_num].file)
            genders.append((d[id_num].gender, d[id_num].patient))
        return files, genders
    testPatientFiles = []
    testLabels = []
    trainPatientFiles = []
    trainLabels = []
    for patient in sessionDict.keys():
        testSessionToAdd = np.random.choice(list(sessionDict[patient].keys()))
        for session in sessionDict[patient].keys():
            files, genders = returnFilesAndLabelsFromSessionDict(sessionDict[patient][session])
            if session == testSessionToAdd:
                testPatientFiles += files
                testLabels += genders
            else:
                trainPatientFiles += files
                trainLabels += genders
    testEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=max_num_samples, edf_tokens=testPatientFiles, n_process=n_process, labels=testLabels)
    testEnsemblerRawData = testEnsembler[:]

    trainEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=max_num_samples, edf_tokens=trainPatientFiles, n_process=n_process, labels=trainLabels)
    trainEnsemblerRawData = trainEnsembler[:]
    trainEnsemblerRawData, validEnsemblerRawData = train_test_split(trainEnsembler, test_size=0.1)

    return trainEnsembler, trainEnsemblerRawData, testEnsembler, testEnsemblerRawData, validEnsemblerRawData

@ex.main
def main(regenerate_data, num_epochs, fit_generator_verbosity, model_name, num_gpus):
    trainDataGen, validDataGen, testDataGen, testEnsemblerRawData = get_data_generator()
    if regenerate_data:
        return
    model = get_model()
    cb_list = get_cb_list()
    history = model.fit_generator(trainDataGen, epochs=num_epochs, validation_data=validDataGen, callbacks=cb_list, verbose=fit_generator_verbosity)

    results = Dict()
    results.history = history.history

    model = keras.models.load_model(model_name)
    model = multi_gpu_model(model, num_gpus)

    y_pred = model.predict(np.stack([data[0].reshape(500,21,1).transpose(1,0,2) for data in testEnsemblerRawData]))

    results.gender.acc = accuracy_score(ypred[0].argmax(1), [data[1][0] for data in testEnsembler])
    results.gender.AUC = roc_auc_score(ypred[0].argmax(1), [data[1][0] for data in testEnsembler])
    results.patient.acc = accuracy_score(ypred[1].argmax(1), [data[1][1] for data in testEnsembler])

    return results


if __name__ == "__main__":
    ex.run_commandline()
