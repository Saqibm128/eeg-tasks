import itertools
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
from os import path
from keras import backend as K
import data_reader as read
import constants
import util_funcs
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator
from keras_models.vanPutten import vp_conv2d, conv2d_gridsearch, inception_like
from keras import optimizers
import pickle as pkl
from sklearn.model_selection import train_test_split
import sacred
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model
from keras_models import train

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
ex = sacred.Experiment(name="predict_patient_in_eeg")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))


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
def standardized_ensemble():
    use_random_ensemble = True
    precached_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_train_data_patient.pkl"
    precached_test_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_test_data_patient.pkl"
    ensemble_sample_info_path = "/n/scratch2/ms994/standardized_edf_ensemble_sample_info_patient.pkl"
    test_train_split_pkl_path = "/n/scratch2/ms994/test_train_split_patient.pkl"
    split_on_sample=True
    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True

all_patients = list(set(read.get_patient_dir_names("combined", "01_tcp_ar", full_path=False)))
all_patients.sort()

@ex.config
def config():
    global all_patients
    train_split = "combined"
    test_split = "combined"
    ref = "01_tcp_ar"
    n_process = 8
    num_files = None
    test_train_split_pkl_path = "test_train_split.pkl"
    max_length = 4 * constants.COMMON_FREQ
    batch_size = 64
    start_offset_seconds = 0  # matters if we aren't doing random ensemble sampling
    dropout = 0.5
    use_early_stopping = True
    patience = 10
    model_name = randomString() + ".h5"
    precached_pkl = "train_data.pkl"
    precached_test_pkl = "test_data.pkl"
    num_epochs = 1000
    lr = 0.002
    total_num_patients = len(all_patients)
    validation_size = 0.2
    test_size = 0.25
    use_cached_pkl = True
    use_vp = True
    # for custom architectures
    num_conv_spatial_layers = 4
    num_conv_temporal_layers = 1
    conv_spatial_filter = (3, 3)
    num_spatial_filter = 100
    conv_temporal_filter = (2, 5)
    num_temporal_filter = 1
    use_filtering = True
    max_pool_size = (1, 3)
    debug_shuffle = False
    max_pool_stride = (1, 2)
    top_k = 5
    use_batch_normalization = True
    use_random_ensemble = False
    max_num_samples = 10  # number of samples of eeg data segments per eeg.edf file
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
    split_on_sample=False


@ex.capture
def get_data_generator(split, pkl_file, total_num_patients, shuffle_generator=True, is_test=False, use_cached_pkl_dataset=True):
    if use_cached_pkl_dataset and path.exists(pkl_file):
        edf = pkl.load(open(pkl_file, 'rb'))
    else:
        edf = EdfDataGenerator(
            get_base_dataset(
                split=split,
                is_test=is_test,
                # edfTokens=testTokens
                ),
            precache=True,
            shuffle=shuffle_generator,
            n_classes=total_num_patients,
            time_first=False
            )
        pkl.dump(edf, open(pkl_file, 'wb'))
    # else:
    #     trainTokens, validTokens, testTokens = get_train_valid_split()
    #     trainEdf = EdfDataGenerator(get_base_dataset(
    #         split=split,
    #         is_test=False,
    #         edfTokens=trainTokens
    #     ),
    #     precache=True,
    #     shuffle=shuffle_generator,
    #     n_classes=total_num_patients,
    #     time_first=False
    #     )
    #     validEdf = EdfDataGenerator(get_base_dataset(
    #         split=split,
    #         is_test=False,
    #         edfTokens=validTokens
    #     ),
    #     precache=True,
    #     shuffle=shuffle_generator,
    #     n_classes=total_num_patients,
    #     time_first=False
    #     )
    #     edf = trainEdf, validEdf
    #     pkl.dump(edf, open(pkl_file, 'wb'))
    return edf

# @ex.capture
# def get_train_valid_split(test_train_split_pkl_path, validation_size, test_size):
#     if not path.exists(test_train_split_pkl_path):
#         edfTokens = read.get_all_token_file_names("combined", "01_tcp_ar")
#         labels = [read.parse_edf_token_path_structure(token)[1] for token in edfTokens]
#         labels = [all_patients.index(label) for label in labels]
#
#         trainValidTokens, testTokens = train_test_split(edfTokens, test_size=test_size)
#         trainTokens, validTokens = train_test_split(trainValidTokens, test_size=validation_size)
#         cached_train_test_split = trainTokens, validTokens, testTokens
#         pkl.dump(cached_train_test_split, open(test_train_split_pkl_path, 'wb'))
#     else:
#         trainTokens, validTokens, testTokens = pkl.load(open(test_train_split_pkl_path, 'rb'))
#     return trainTokens, validTokens, testTokens

@ex.capture
def get_base_dataset(split,
                     ref,
                     n_process,
                     num_files,
                     max_num_samples,
                     max_length,
                     start_offset_seconds,
                     ensemble_sample_info_path,
                     use_standard_scaler,
                     use_filtering,
                     edfTokens = None,
                     use_cached_pkl_dataset=True,
                     is_test=False,
                     is_valid=False):
    if edfTokens is None:
        edfTokens = read.get_all_token_file_names(split, ref)
    patients = [read.parse_edf_token_path_structure(file)[1] for file in edfTokens]
    global all_patients
    patients = [all_patients.index(patient) for patient in patients]
    edfData = er.EdfDatasetEnsembler(
        split,
        ref,
        n_process=n_process,
        edf_tokens=edfTokens,
        labels=patients,
        max_length=max_length * pd.Timedelta(seconds=constants.COMMON_DELTA),
        use_numpy=True,
        num_files=num_files,
        max_num_samples=max_num_samples,
        filter=use_filtering
    )
    samplingInfo = edfData.sampleInfo
    if use_standard_scaler:
        edfData = read.EdfStandardScaler(
            edfData, dataset_includes_label=True, n_process=n_process)
    basename = path.basename(ensemble_sample_info_path)
    dirname = path.dirname(ensemble_sample_info_path)
    ensemble_sample_info_path = (dirname + "/test_" + \
        basename) if is_test else ensemble_sample_info_path
    ensemble_sample_info_path = (dirname + "/valid_" + \
        basename) if is_valid else ensemble_sample_info_path
    if use_cached_pkl_dataset and path.exists(ensemble_sample_info_path):
        edfData.sampleInfo = pkl.load(
            open(ensemble_sample_info_path, 'rb'))
        ex.add_resource(ensemble_sample_info_path)
    else:
        pkl.dump(samplingInfo, open(ensemble_sample_info_path, 'wb'))
        ex.add_artifact(ensemble_sample_info_path)
    edfData.verbosity = 50
    return edfData


@ex.capture
def get_model(dropout, max_length, lr, use_vp, top_k, num_spatial_filter, use_batch_normalization, max_pool_size, use_inception_like, total_num_patients, output_activation="softmax", num_gpus=1, num_outputs=None,):
    if num_outputs is None:
        num_outputs = total_num_patients
    if use_vp:
        model = vp_conv2d(
            dropout=dropout,
            input_shape=(21, max_length, 1),
            filter_size=num_spatial_filter,
            max_pool_size=max_pool_size,
            use_batch_normalization=use_batch_normalization,
            output_activation=output_activation,
            num_outputs=num_outputs
            )

    elif use_inception_like:
        model = get_inception_like(
            output_activation=output_activation,
            num_outputs=num_outputs)
    else:
        model = get_custom_model(
            output_activation=output_activation,
            num_outputs=num_outputs)
    model.summary()
    if num_gpus != 1:
        model = multi_gpu_model(model, num_gpus)
    adam = optimizers.Adam(lr=lr)
    def top_k_categorical_accuracy(x, y):
        x = K.eval(x)
        y = K.eval(y)
        return K.cast(np.mean([x.argmax(1)[i] in y.argsort(1)[:,-top_k:][i] for i in range(len(x))]))
    model.compile(adam, loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    return model

@ex.capture
def get_inception_like(max_length, num_conv_spatial_layers, num_spatial_filter, dropout, lr, output_activation='softmax', num_outputs=2):
    model = inception_like((21, max_length, 1), num_layers=num_conv_spatial_layers, num_filters=num_spatial_filter, dropout=dropout, output_activation=output_activation, num_outputs=num_outputs)
    return model


@ex.capture
def get_custom_model(
    dropout,
    max_length,
    lr,
    use_batch_normalization,
    num_conv_spatial_layers=1,
    num_conv_temporal_layers=1,
    conv_spatial_filter=(3, 3),
    num_spatial_filter=100,
    conv_temporal_filter=(2, 5),
    num_temporal_filter=300,
    max_pool_size=(2, 2),
    max_pool_stride=(1, 2),
    output_activation='softmax',
    num_outputs=2
):
    model = conv2d_gridsearch(
        dropout=dropout,
        input_shape=(21, max_length, 1),
        num_conv_spatial_layers=num_conv_spatial_layers,
        num_conv_temporal_layers=num_conv_temporal_layers,
        conv_spatial_filter=conv_spatial_filter,
        num_spatial_filter=num_spatial_filter,
        conv_temporal_filter=conv_temporal_filter,
        num_temporal_filter=num_temporal_filter,
        max_pool_size=max_pool_size,
        max_pool_stride=max_pool_stride,
        use_batch_normalization=use_batch_normalization,
        output_activation=output_activation,
        num_outputs=num_outputs
    )
    return model



@ex.main
def main(use_dl):
    # if use_dl:
    return dl()  # deep learning branch

@ex.capture
def get_train_valid_generator(train_split, precached_pkl):
    return get_data_generator(train_split, pkl_file=precached_pkl, is_test=False)

@ex.capture
def get_test_generator(test_split, precached_test_pkl):
    return get_data_generator(test_split, pkl_file=precached_test_pkl, shuffle_generator=False, is_test=True)


@ex.capture
def get_model_checkpoint(model_name, monitor='val_loss'):
    return ModelCheckpoint(model_name, monitor=monitor, save_best_only=True, verbose=1, mode="min")


@ex.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on, mode="max")


@ex.capture
def get_cb_list(use_early_stopping, model_name):
    if use_early_stopping:
        return [get_early_stopping(), get_model_checkpoint(),]
    else:
        return [get_model_checkpoint()]

@ex.capture
def get_train_valid_test_data_get_session(precached_pkl, shuffle_generator, use_cached_pkl_dataset=True):
    trainTokens, validTokens, testTokens = get_train_valid_test_split_session()
    if use_cached_pkl_dataset and path.exists(precached_pkl):
        edf = pkl.load(open(precached_pkl, 'rb'))
    else:
        global all_patients
        total_num_patients = len(all_patients)
        split="combined"
        trainEdf = EdfDataGenerator(
            get_base_dataset(
                split=split,
                is_test=False,
                edfTokens=trainTokens
                ),
            precache=True,
            shuffle=shuffle_generator,
            n_classes=total_num_patients,
            time_first=False
            )
        validEdf = EdfDataGenerator(
            get_base_dataset(
                split=split,
                is_test=False,
                is_valid=True,
                edfTokens=validTokens
                ),
            precache=True,
            shuffle=shuffle_generator,
            n_classes=total_num_patients,
            time_first=False
            )
        testEdf = EdfDataGenerator(
            get_base_dataset(
                split=split,
                is_test=True,
                edfTokens=testTokens
                ),
            precache=True,
            shuffle=False,
            n_classes=total_num_patients,
            time_first=False
            )
        edf = trainEdf, validEdf, testEdf
        pkl.dump(edf, open(precached_pkl,'wb'))
    return edf

@ex.capture
def get_train_valid_test_split_session(test_train_split_pkl_path, validation_size=0.2, test_size=0.25):
    if not path.exists(test_train_split_pkl_path):
        patients = read.get_patient_dir_names("combined", "01_tcp_ar")
        patientToSess = Dict()
        sessions  = []
        labels = []
        for patientFile in patients:
            patientToSess[patientFile] = read.get_session_dir_names("combined", "01_tcp_ar", patient_dirs=[patientFile])
            if len(patientToSess[patientFile]) < 2: #need at least 2 to do a train_valid_test split
                del patientToSess[patientFile]
            else:
                sessions += (patientToSess[patientFile])
                labels += [patientFile for i in range(len(patientToSess[patientFile]))]

        test_size = (len(set(labels)) + 1) / len(sessions) #grab all of the labels to put in test set!
        trainValidSessions, testSessions, trainValidLabels, testLabels = train_test_split(sessions, labels, stratify=labels, test_size=test_size)

        testEdfTokens = list(itertools.chain.from_iterable([read.get_token_file_names(session) for session in testSessions]))
        trainValidEdfTokens = list(itertools.chain.from_iterable([read.get_token_file_names(session) for session in trainValidSessions]))
        trainEdfTokens, validEdfTokens = train_test_split(trainValidEdfTokens, test_size=validation_size)
        pkl.dump((trainEdfTokens, validEdfTokens, testEdfTokens), open(test_train_split_pkl_path, 'wb'))
    else:
        trainEdfTokens, validEdfTokens, testEdfTokens = pkl.load(open(test_train_split_pkl_path, 'rb'))

    return trainEdfTokens, validEdfTokens, testEdfTokens

@ex.capture
def dl(
    train_split,
    test_split,
    num_epochs,
    lr,
    top_k,
    batch_size,
    n_process,
    validation_size,
    test_size,
    max_length,
    total_num_patients,
    use_random_ensemble,
    ref,
    num_files,
    regenerate_data,
    model_name,
    use_standard_scaler,
    fit_generator_verbosity,
    validation_steps,
    steps_per_epoch,
    num_gpus,
    ensemble_sample_info_path,
    split_on_sample,
    debug_shuffle = False
    ):
    if not split_on_sample:
        data = get_train_valid_generator()
        trainDataGenerator, testDataGenerator = data.create_validation_train_split(test_size)
        trainDataGenerator, validDataGenerator = data.create_validation_train_split(validation_size)
        pkl.dump(testDataGenerator, open("/n/scratch2/ms994/" + model_name + ".train_data.pkl", 'wb')) #way to force replicate test set
    else:
        trainDataGenerator, validDataGenerator, testDataGenerator = get_train_valid_test_data_get_session()
    # trainValidationDataGenerator.time_first = False
    # testDataGenerator = get_test_generator()
    # trainValidationDataGenerator.n_classes=2
    # trainDataGenerator, validDataGenerator = trainValidationDataGenerator.create_validation_train_split(validation_size)
    if regenerate_data:
        return

    # return
    model = get_model()

    best_val_score = -100
    trainDataGenerator.batch_size = batch_size
    validDataGenerator.batch_size = batch_size


    history = model.fit_generator(
            trainDataGenerator,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            validation_data = validDataGenerator,
            verbose=fit_generator_verbosity,
            epochs=num_epochs,
            callbacks=get_cb_list(),
            )



    testData = testDataGenerator.dataset
    testData = np.stack(np.stack(testData)[:,0])
    testData = testData.reshape((*testData.shape, 1)).transpose(0,2,1,3)
    testEdfEnsembler = er.EdfDatasetEnsembler("dev_test", ref, generate_sample_info=False)
    testDataGenerator.shuffle=False
    testDataGenerator.batch_size = 256*num_gpus
    basename = path.basename(ensemble_sample_info_path)
    dirname = path.dirname(ensemble_sample_info_path)
    # test_ensemble_sample_info_path = (dirname + "/test_" + \
        # basename)
    # testEdfEnsembler.sampleInfo = pkl.load(open(test_ensemble_sample_info_path, 'rb'))
    # testPatient = testEdfEnsembler.getEnsembledLabels()

    model = keras.models.load_model(model_name)
    if num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)


    y_pred = model.predict_generator(testDataGenerator) #time second, feature first
    # y_pred = y_pred.reshape((-1, *y_pred[-2:]))
    testPatients = []
    for i in range(len(testDataGenerator)):
        testData, toAppend = testDataGenerator[i]
        testPatients.append(toAppend.argmax(1))
    testPatients = np.hstack(testPatients)

    print("pred shape", y_pred.shape)
    print("test data shape", testData.shape)

    try:
        auc = roc_auc_score(testPatients, y_pred.argmax(axis=1))
    except Exception:
        print("Could not calculate auc")
        auc=0.5
    f1 = f1_score(testPatients, y_pred.argmax(axis=1), average='weighted')

    def top_k_acc(k):
        top_k_pred = y_pred.argsort()[:,-(k):]
        return np.mean([testPatients[i] in top_k_pred[i] for i in range(len(testPatients))])

    k_accuracy = top_k_acc(top_k)
    multi_k_acc = {
        "1":top_k_acc(1),
        "2":top_k_acc(2),
        "5":top_k_acc(5),
        "10":top_k_acc(10),
        "20":top_k_acc(20)
    }

    accuracy = accuracy_score(testPatients, y_pred.argmax(1))

    toReturn = {
        'history': history.history,
        'val_scores': {
            'min_val_loss': min(history.history['val_loss']),
        },
        'test_scores': {
            'f1': f1,
            'k_acc': k_accuracy,
            'acc': accuracy,
            'auc': auc,
            'multi_k_acc':multi_k_acc
        }}

    return toReturn



if __name__ == "__main__":
    ex.run_commandline()
