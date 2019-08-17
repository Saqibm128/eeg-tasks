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
from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like
from keras import losses
from keras import optimizers
import pickle as pkl
import sacred
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
ex = sacred.Experiment(name="predict_age_conv")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))



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
def standardized_ensemble():
    use_random_ensemble = True
    precached_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_train_data_age.pkl"
    precached_test_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_test_data_age.pkl"
    ensemble_sample_info_path = "/n/scratch2/ms994/standardized_edf_ensemble_sample_info_age.pkl"

    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True
    standardize_ages = 100

@ex.named_config
def standardized_ensemble_5():
    use_random_ensemble = True
    precached_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_train_data_age5.pkl"
    precached_test_pkl = "/n/scratch2/ms994/standardized_simple_ensemble_test_data_age5.pkl"
    ensemble_sample_info_path = "/n/scratch2/ms994/standardized_edf_ensemble_sample_info_age5.pkl"

    max_num_samples = 5  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True
    standardize_ages = 100



@ex.named_config
def stop_on_training_loss():
    early_stopping_on = "mean_squared_error"

@ex.config
def config():
    train_split = "train"
    test_split = "dev_test"
    ref = "01_tcp_ar"
    n_process = 8
    num_files = None
    max_length = 4 * constants.COMMON_FREQ
    batch_size = 64
    start_offset_seconds = 0  # matters if we aren't doing random ensemble sampling
    dropout = 0.25
    use_early_stopping = True
    patience = 10
    model_name = randomString() + ".h5"
    precached_pkl = "train_data.pkl"
    precached_test_pkl = "test_data.pkl"
    num_epochs = 1000
    lr = 0.0002
    validation_size = 0.2
    test_size = 0.2
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
    max_pool_stride = (1, 2)
    regenerate_data = False
    use_batch_normalization = True
    use_random_ensemble = False
    max_num_samples = 10  # number of samples of eeg data segments per eeg.edf file
    num_gpus = 1
    early_stopping_on = "val_mean_squared_error"
    test_train_split_pkl_path = "train_test_split_info.pkl"
    # if use_cached_pkl is false and this is true, just generates pickle files, doesn't make models or anything
    use_standard_scaler = False
    ensemble_sample_info_path = "edf_ensemble_path.pkl"
    fit_generator_verbosity = 2
    steps_per_epoch = None
    validation_steps = None
    shuffle_generator = True
    use_dl = True
    use_inception_like=False
    continue_from_model=False
    standardize_ages=None
    output_activation = "linear"


# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))


@ex.capture
def get_model_checkpoint(model_name, monitor='val_mean_squared_error'):
    return ModelCheckpoint(model_name, monitor=monitor, save_best_only=True, verbose=1)


@ex.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on, mode="min")


@ex.capture
def get_cb_list(use_early_stopping, model_name):
    if use_early_stopping:
        return [get_early_stopping(), get_model_checkpoint(),]
    else:
        return [get_model_checkpoint()]

@ex.capture
def get_data_generator(split, pkl_file, shuffle_generator, is_test=False, use_cached_pkl_dataset=True):
    if use_cached_pkl_dataset and path.exists(pkl_file):
        edf = pkl.load(open(pkl_file, 'rb'))
    elif is_test:
        edf = EdfDataGenerator(
            get_base_dataset(
                split=split,
                is_test=is_test
                ),
            precache=True,
            shuffle=shuffle_generator,
            class_type="quantile",
            time_first=False
            )
        pkl.dump(edf, open(pkl_file, 'wb'))
    else:
        trainTokens, validTokens = get_train_valid_split()
        trainEdf = EdfDataGenerator(get_base_dataset(
            split=split,
            is_test=False,
            edf_tokens=trainTokens
        ),
        precache=True,
        shuffle=shuffle_generator,
        class_type="quantile",
        n_classes=2,
        time_first=False
        )
        validEdf = EdfDataGenerator(get_base_dataset(
            split=split,
            is_test=False,
            edf_tokens=validTokens
        ),
        class_type="quantile",
        precache=True,
        shuffle=shuffle_generator,
        n_classes=2,
        time_first=False
        )
        edf = trainEdf, validEdf
        pkl.dump(edf, open(pkl_file, 'wb'))
    return edf

cached_train_test_split = None
@ex.capture
def get_train_valid_split():
    global cached_train_test_split
    if cached_train_test_split is None:
        edfTokens, ages = cta.demux_to_tokens(cta.getAgesAndFileNames("train", "01_tcp_ar"))
        trainTokens, validTokens, _a, _b = cta.train_test_split_on_combined(edfTokens, edfTokens, 0.1, stratify=False)
        cached_train_test_split = trainTokens, validTokens
    else:
        trainTokens, validTokens = cached_train_test_split
    return trainTokens, validTokens

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
                     edf_tokens=None,
                     use_cached_pkl_dataset=True,
                     is_test=False,
                     is_valid=False,
                     standardize_ages=None
                     ):

    files, ages = cta.demux_to_tokens(cta.getAgesAndFileNames(split, ref))
    ages = np.stack(ages)
    if standardize_ages is not None:
        ages = (ages - standardize_ages/2) / standardize_ages
    if edf_tokens is not None:
        ageDict = {}
        for i, file in enumerate(files):
            ageDict[file] = ages[i]
        ages = [ageDict[file] for file in edf_tokens]
        files = edf_tokens
    edfData = er.EdfDatasetEnsembler(
        split,
        ref,
        edf_tokens=files,
        labels=ages,
        n_process=n_process,
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
def load_saved_model(model_name, num_gpus):
    model = keras.models.load_model(model_name)
    if num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)
    return model

@ex.capture
def get_model(dropout, max_length, continue_from_model, lr, use_vp, num_spatial_filter, use_batch_normalization, max_pool_size, use_inception_like, output_activation="linear", num_gpus=1, num_outputs=1):
    if continue_from_model:
        model = load_saved_model()
    elif use_vp:
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
            num_outputs=num_outputs,
            )
    else:
        model = get_custom_model(
            output_activation=output_activation,
            num_outputs=num_outputs
            )
    model.summary()
    if num_gpus != 1:
        model = multi_gpu_model(model, num_gpus)
    adam = optimizers.Adam(lr=lr)
    model.compile(adam, loss=losses.mean_squared_error,
                  metrics=["mean_absolute_error", "mean_squared_error"])
    return model

@ex.capture
def get_inception_like(max_length, num_conv_spatial_layers, num_spatial_filter, dropout, lr, output_activation, num_outputs=2):
    model = inception_like((21, max_length, 1), num_layers=num_conv_spatial_layers, num_filters=num_spatial_filter, dropout=dropout, output_activation=output_activation, num_outputs=num_outputs)
    adam = optimizers.Adam(lr=lr)
    return model


@ex.capture
def get_custom_model(
    dropout,
    max_length,
    lr,
    output_activation,
    use_batch_normalization,
    num_conv_spatial_layers=1,
    num_conv_temporal_layers=1,
    conv_spatial_filter=(3, 3),
    num_spatial_filter=100,
    conv_temporal_filter=(2, 5),
    num_temporal_filter=300,
    max_pool_size=(2, 2),
    max_pool_stride=(1, 2),
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
    adam = optimizers.Adam(lr=lr)
    return model



@ex.main
def main(use_dl):
    return dl()  # deep learning branch

@ex.capture
def get_train_valid_generator(train_split, precached_pkl):
    return get_data_generator(train_split, pkl_file=precached_pkl, is_test=False)

@ex.capture
def get_test_generator(test_split, precached_test_pkl):
    return get_data_generator(test_split, pkl_file=precached_test_pkl, shuffle_generator=False, is_test=True)

@ex.capture
def dl(train_split, test_split, num_epochs, lr, n_process, validation_size, max_length, use_random_ensemble, ref, num_files, regenerate_data, model_name, use_standard_scaler, fit_generator_verbosity, validation_steps, steps_per_epoch, num_gpus, ensemble_sample_info_path):
    trainDataGenerator, validDataGenerator = get_train_valid_generator()
    # trainValidationDataGenerator.time_first = False
    testDataGenerator = get_test_generator()
    # trainValidationDataGenerator.n_classes=2
    # trainDataGenerator, validDataGenerator = trainValidationDataGenerator.create_validation_train_split(validation_size)
    if regenerate_data:
        return
    # return
    model = get_model()
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
    basename = path.basename(ensemble_sample_info_path)
    dirname = path.dirname(ensemble_sample_info_path)
    test_ensemble_sample_info_path = (dirname + "/test_" + \
        basename)
    testEdfEnsembler.sampleInfo = pkl.load(open(test_ensemble_sample_info_path, 'rb'))
    if use_standard_scaler:
        testGender = testEdfEnsembler.getEnsembledLabels()
    else:
        testGender = testEdfEnsembler.getEnsembledLabels()

    model = load_saved_model()


    y_pred = model.predict(testData) #time second, feature first
    print("pred shape", y_pred.shape)
    print("test data shape", testData.shape)

    mse = mean_squared_error(testGender, y_pred)
    r2 = r2_score(testGender, y_pred)

    toReturn = {
        'history': history.history,
        'val_scores': {
            'min_val_mse': min(history.history['val_mean_squared_error']),
        },
        'test_scores': {
            'r2_score': r2,
            'mse': mse,
        },}




    label, average_pred = testEdfEnsembler.getEnsemblePrediction(
        y_pred, mode=er.EdfDatasetEnsembler.ENSEMBLE_PREDICTION_EQUAL_VOTE)
    label, average_over_all_pred = testEdfEnsembler.getEnsemblePrediction(
        y_pred, mode=er.EdfDatasetEnsembler.ENSEMBLE_PREDICTION_OVER_EACH_SAMP)
    pred = np.round(average_pred)
    toReturn["ensemble_score"] = {
        "equal_vote":{},
        "over_all":{}
    }

    toReturn["ensemble_score"]["equal_vote"]["r2"] = r2_score(label, pred)
    toReturn["ensemble_score"]["equal_vote"]["mse"] = mean_squared_error(label, pred)
    toReturn["ensemble_score"]["equal_vote"]["discordance"] = np.abs(
        pred - average_pred).mean()

    pred = np.round(average_over_all_pred)
    toReturn["ensemble_score"]["over_all"]["r2"] = r2_score(label, pred)
    toReturn["ensemble_score"]["over_all"]["mse"] = mean_squared_error(label, pred)
    toReturn["ensemble_score"]["over_all"]["discordance"] = np.abs(
        pred - average_pred).mean()

    return toReturn



if __name__ == "__main__":
    ex.run_commandline()
