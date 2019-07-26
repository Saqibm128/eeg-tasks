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
from keras_models.dataGen import EdfDataGenerator
from keras_models.vanPutten import vp_conv2d, conv2d_gridsearch
from keras import optimizers
import pickle as pkl
import sacred
import keras

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping
ex = sacred.Experiment(name="gender_predict_conv_gridsearch")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

# trainEdfEnsembler = None
# testEdfEnsembler = None


@ex.named_config
def rf():
    use_dl = False
    max_train_rf_samps = None
    freq_bins = [0, 10, 20, 25, 27.5, 30]
    rf_data_pickle = "rf_fft_data.pkl"


@ex.named_config
def conv_spatial_filter_2_2():
    conv_spatial_filter = (2, 2)


@ex.named_config
def conv_spatial_filter_3_3():
    conv_spatial_filter = (3, 3)


@ex.named_config
def conv_temporal_filter_1_7():
    conv_temporal_filter = (1, 7)

@ex.named_config
def conv_temporal_filter_1_3():
    conv_temporal_filter = (1, 3)


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
def combined_less_data_simple_ensemble_samples():
    use_random_ensemble = True
    precached_pkl = "combined_simple_ensemble_train_data_40_segs_max_length_2.pkl"
    precached_test_pkl = "combined_simple_ensemble_test_data_40_segs_max_length_2.pkl"
    ensemble_sample_info_path = "2s_edf_ensemble_path.pkl"
    max_length = 2 * constants.COMMON_FREQ
    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file


@ex.named_config
def simple_ensemble():
    '''
    Not really training as an ensemble, except for the test phase, when we try to see our stats as an ensemble
    '''
    use_random_ensemble = True
    precached_pkl = "simple_ensemble_train_data_max_length_4.pkl"
    precached_test_pkl = "simple_ensemble_test_data_max_length_4.pkl"
    ensemble_sample_info_path = "native_edf_ensemble_path.pkl"
    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file


@ex.named_config
def standardized_combined_simple_ensemble():
    use_combined = True
    use_random_ensemble = True
    train_split = "combined"
    test_split = "combined"
    precached_pkl = "standardized_combined_simple_ensemble_train_data.pkl"
    precached_test_pkl = "standardized_combined_simple_ensemble_test_data.pkl"
    ensemble_sample_info_path = "standardized_edf_ensemble_sample_info.pkl"

    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = True
    use_filtering = True


@ex.named_config
def combined_simple_ensemble():
    use_combined = True
    use_random_ensemble = True
    train_split = "combined"
    test_split = "combined"
    precached_pkl = "combined_simple_ensemble_train_data.pkl"
    precached_test_pkl = "combined_simple_ensemble_test_data.pkl"

    max_num_samples = 40  # number of samples of eeg data segments per eeg.edf file
    use_standard_scaler = False


@ex.named_config
def combined():
    use_combined = True
    precached_pkl = "combined_train_data.pkl"
    precached_test_pkl = "combined_test_data.pkl"


@ex.named_config
def stop_on_training_loss():
    early_stopping_on = "loss"


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
    normalize_inputs = False
    # for custom architectures
    num_conv_spatial_layers = 1
    num_conv_temporal_layers = 1
    conv_spatial_filter = (3, 3)
    num_spatial_filter = 100
    conv_temporal_filter = (2, 5)
    num_temporal_filter = 1
    use_filtering = True
    max_pool_size = (2, 2)
    max_pool_stride = (1, 2)
    use_batch_normalization = True
    use_random_ensemble = False
    max_num_samples = 10  # number of samples of eeg data segments per eeg.edf file
    use_combined = False
    combined_split = "combined"
    early_stopping_on = "val_loss"
    test_train_split_pkl_path = "train_test_split_info.pkl"
    # if use_cached_pkl is false and this is true, just generates pickle files, doesn't make models or anything
    regenerate_data = False
    use_standard_scaler = False
    ensemble_sample_info_path = "edf_ensemble_path.pkl"
    fit_generator_verbosity = 2
    steps_per_epoch = None
    shuffle_generator = True
    use_dl = True


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
    return EarlyStopping(patience=patience, verbose=1)


@ex.capture
def get_cb_list(use_early_stopping, model_name):
    if use_early_stopping:
        return [get_early_stopping(), get_model_checkpoint(), get_model_checkpoint("bin_acc_"+model_name, "val_binary_accuracy")]
    else:
        return [get_model_checkpoint(), get_model_checkpoint("bin_acc_"+model_name, "val_binary_accuracy")]


@ex.capture
def get_base_dataset(split,
                     ref,
                     n_process,
                     num_files,
                     use_random_ensemble,
                     labels,
                     max_num_samples,
                     max_length,
                     edfTokenPaths,
                     start_offset_seconds,
                     ensemble_sample_info_path,
                     use_standard_scaler,
                     use_filtering,
                     use_cached_pkl_dataset=True,
                     is_test=False,
                     is_valid=False):
    if not use_random_ensemble:
        edfData = read.EdfDataset(
            split,
            ref,
            n_process=n_process,
            max_length=max_length *
            pd.Timedelta(seconds=constants.COMMON_DELTA),
            use_numpy=True,
            start_offset=pd.Timedelta(seconds=start_offset_seconds),
            filter=use_filtering
        )
        edfData.edf_tokens = edfTokenPaths[:num_files]
        if use_standard_scaler:
            edfData = read.EdfStandardScaler(
                edfData, dataset_includes_label=True)
        assert len(edfData) == len(labels)
        return edfData
    else:  # store the ensemble data and the info on how stuff was sampled out
        edfData = read.EdfDatasetEnsembler(
            split,
            ref,
            n_process=n_process,
            max_length=max_length *
            pd.Timedelta(seconds=constants.COMMON_DELTA),
            use_numpy=True,
            edf_tokens=edfTokenPaths[:num_files],
            labels=labels[:num_files],
            max_num_samples=max_num_samples,
            filter=use_filtering
        )
        samplingInfo = edfData.sampleInfo
        if use_standard_scaler:
            edfData = read.EdfStandardScaler(
                edfData, dataset_includes_label=True, n_process=n_process)
        ensemble_sample_info_path = "test_" + \
            ensemble_sample_info_path if is_test else ensemble_sample_info_path
        ensemble_sample_info_path = "valid_" + \
            ensemble_sample_info_path if is_valid else ensemble_sample_info_path
        print(ensemble_sample_info_path)
        if use_cached_pkl_dataset and path.exists(ensemble_sample_info_path):
            edfData.sampleInfo = pkl.load(
                open(ensemble_sample_info_path, 'rb'))
            ex.add_resource(ensemble_sample_info_path)
        else:
            pkl.dump(samplingInfo, open(ensemble_sample_info_path, 'wb'))
            ex.add_artifact(ensemble_sample_info_path)
        edfData.verbosity = 50
        return edfData


cached_test_train_split_info = None
@ex.capture
def get_test_train_split_from_combined(combined_split, ref, test_size,  test_train_split_pkl_path, validation_size, use_cached_pkl_test_train_split=True,):
    global cached_test_train_split_info
    if use_cached_pkl_test_train_split and path.exists(test_train_split_pkl_path):
        cached_test_train_split_info = pkl.load(
            open(test_train_split_pkl_path, 'rb'))
        ex.add_resource(test_train_split_pkl_path)
    elif cached_test_train_split_info is None:
        edfTokens, genders = cta.demux_to_tokens(cta.getGenderAndFileNames(
            combined_split, ref, convert_gender_to_num=True))
        trainTokens, testTokens, trainGenders, testGenders = cta.train_test_split_on_combined(
            edfTokens, genders, test_size=test_size)
        trainTokens, validationTokens, trainGenders, validationGenders = cta.train_test_split_on_combined(
            trainTokens, trainGenders, test_size=validation_size)
        assert len(set(trainTokens).intersection(validationTokens)) == 0
        assert len(set(trainTokens).intersection(testTokens)) == 0

        cached_test_train_split_info = trainTokens, validationTokens, testTokens, trainGenders, validationGenders, testGenders
        pkl.dump(cached_test_train_split_info, open(
            test_train_split_pkl_path, 'wb'))
        ex.add_artifact(test_train_split_pkl_path)
    # trainData, testData, trainGender, testGender
    return cached_test_train_split_info


@ex.capture
def get_data(split, ref, n_process, num_files, max_length, precached_pkl, precached_test_pkl, use_cached_pkl_dont_reload_data=True, use_combined=False, train_split="", test_split="",  is_test=False, is_valid=False):
    if is_test:
        precached_pkl = precached_test_pkl
    if use_combined:  # use the previous test train split, since we are sharing a split instead of enforcing it with a different directory
        edfTokensTrain, edfTokensValidation, edfTokensTest, gendersTrain, gendersValidation, gendersTest = get_test_train_split_from_combined()
        if split == train_split and not is_test and not is_valid:
            edfTokenPaths = edfTokensTrain[:num_files]
            genders = np.array(gendersTrain[:num_files])
            assert len(edfTokenPaths) == len(genders)
        elif is_test:
            edfTokenPaths = edfTokensTest[:num_files]
            genders = np.array(gendersTest[:num_files])
        elif is_valid:
            edfTokenPaths = edfTokensValidation[:num_files]
            genders = np.array(gendersValidation[:num_files])
    else:
        genderDict = cta.getGenderAndFileNames(
            split, ref, convert_gender_to_num=True)
        edfTokenPaths, genders = cta.demux_to_tokens(genderDict)
    if is_valid:
        precached_pkl = "valid_" + precached_pkl
    if path.exists(precached_pkl) and use_cached_pkl_dont_reload_data:
        edfData = pkl.load(open(precached_pkl, 'rb'))
    else:
        edfData = get_base_dataset(
            split, labels=genders, edfTokenPaths=edfTokenPaths, is_test=is_test, is_valid=is_valid)
        edfData = edfData[:]
        # don't add these as artifacts or resources or else mongodb will try to store giant file copies of these
        pkl.dump(edfData, open(precached_pkl, 'wb'))
    return edfData, genders


@ex.capture
def get_data_generator(split, batch_size, num_files, max_length, use_random_ensemble, shuffle_generator, split_type="", ):
    """Based on a really naive, dumb mapping of eeg electrodes into 2d space

    Parameters
    ----------
    split : type
        Description of parameter `split`.
    ref : type
        Description of parameter `ref`.
    n_process : type
        Description of parameter `n_process`.

    Returns
    -------
    type
        Description of returned object.

    """
    edfData, genders = get_data(split, is_test=split_type == "test", is_valid=(
        split_type == "validation" or split_type == "valid"))
    return EdfDataGenerator(
        edfData,
        precache=True,
        time_first=False,
        n_classes=2,
        # properly duplicated genders inside edfData if using use_random_ensemble
        labels=np.array(genders) if not use_random_ensemble else None,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle_generator
    )


@ex.capture
def get_model(dropout, max_length, lr, use_vp, num_spatial_filter, use_batch_normalization):
    if use_vp:
        model = vp_conv2d(dropout=dropout, input_shape=(21, max_length, 1),
                          filter_size=num_spatial_filter, use_batch_normalization=use_batch_normalization)
        adam = optimizers.Adam(lr=lr)
        model.compile(adam, loss="categorical_crossentropy",
                      metrics=["binary_accuracy"])
        return model
    else:
        return get_custom_model()


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
        use_batch_normalization=use_batch_normalization
    )
    adam = optimizers.Adam(lr=lr)
    model.compile(adam, loss="categorical_crossentropy",
                  metrics=["binary_accuracy"])
    return model


@ex.capture
def get_test_data(test_split, max_length, precached_test_pkl, use_cached_pkl):
    testData, testGender = get_data(
        test_split, precached_pkl=precached_test_pkl, is_test=True)
    testData = np.stack([datum[0] for datum in testData])
    testData = testData[:, 0:max_length]
    testData = testData.reshape(*testData.shape, 1).transpose(0, 2, 1, 3)
    return testData, testGender


@ex.main
def main(use_dl):
    if use_dl:
        return dl()  # deep learning branch
    else:
        return run_rf()


def split_data_gender(dataGender):
    """ For rearranging data in the form of [(datum, gender),...,(datum, gender)]
    into np.arrays (list of tuples is hard to use)

    Parameters
    ----------
    dataGender : list
        List of tuples

    Returns
    -------
    type
        Description of returned object.

    """
    # did data in n by 2 data (I HATE MYSELF), this returns neat and correct arrays of data.
    data = np.stack([datum[0] for datum in dataGender])
    gender = np.array([datum[1] for datum in dataGender])
    return data, gender


@ex.capture
def run_rf(use_combined, use_random_ensemble, combined_split, freq_bins, max_train_rf_samps, n_process, rf_data_pickle, use_cached_pkl):
    if not use_combined or not use_random_ensemble:
        raise NotImplemented("Have not completed this flow yet")
    else:
        if not use_cached_pkl or not path.exists(rf_data_pickle):
            trainEdfData, trainGender = split_data_gender(get_data(
                combined_split, is_test=False, is_valid=False)[0][:max_train_rf_samps])
            trainEdfData = read.EdfFFTDatasetTransformer(
                trainEdfData, return_numpy=True, is_tuple_data=False, is_pandas_data=False, freq_bins=freq_bins, n_process=n_process)
            trainEdfData.verbosity = 200
            trainEdfData = trainEdfData[:]
            validEdfData, validGender = split_data_gender(
                get_data(combined_split, is_test=False, is_valid=True)[0][:max_train_rf_samps])
            validEdfData = read.EdfFFTDatasetTransformer(
                validEdfData, return_numpy=True, is_tuple_data=False, is_pandas_data=False, freq_bins=freq_bins, n_process=n_process)
            validEdfData.verbosity = 200
            validEdfData = validEdfData[:]

            trainSize = len(trainEdfData)
            validSize = len(validEdfData)
            trainValidData = np.vstack(
                [np.stack(trainEdfData), np.stack(validEdfData)])
            trainValidData = trainValidData.reshape(
                trainValidData.shape[0], -1)
            trainValidGender = np.hstack(
                [np.array(trainGender), np.array(validGender)]).reshape(-1, 1)
            # deallocate memory so o2 doesn't kick this out when we try to start training, etc.
            del trainEdfData
            del validEdfData
            pkl.dump((trainSize, validSize, trainValidData,
                      trainValidGender), open(rf_data_pickle, 'wb'))
        else:
            trainSize, validSize, trainValidData, trainValidGender = pkl.load(
                open(rf_data_pickle, 'rb'))

        rf = RandomForestClassifier()
        preSplit = PredefinedSplit(
            [0 for i in range(trainSize)] + [-1 for i in range(validSize)])
        rf_parameters = {
            'criterion': ["gini", "entropy"],
            'n_estimators': [50, 100, 200, 400],
            'max_features': ['auto', 'log2', .1, .4],
            'max_depth': [None,  4, 8, 12],
            'min_samples_split': [2, 4, 8],
            'n_jobs': [1],
            'min_weight_fraction_leaf': [0, 0.2]
        }
        gridsearch = GridSearchCV(rf, rf_parameters, scoring=make_scorer(
            f1_score), cv=preSplit, n_jobs=n_process)
        gridsearch.fit(trainValidData, trainValidGender)


        if not use_cached_pkl or not path.exists("test_" + rf_data_pickle):
            testEdfData, testGender = split_data_gender(
                get_data(combined_split, is_test=True, is_valid=False)[0][:max_train_rf_samps])
            testEdfData = read.EdfFFTDatasetTransformer(
                testEdfData, return_numpy=True, is_tuple_data=False, is_pandas_data=False, freq_bins=freq_bins, n_process=n_process)
            testEdfData.verbosity = 200
            testEdfData = testEdfData[:]
            pkl.dump((testEdfData, testGender), open(
                "test_" + rf_data_pickle, 'wb'))
        else:
            testEdfData, testGender = pkl.load(
                open("test_" + rf_data_pickle, 'rb'))

        y_pred = gridsearch.predict(
            np.stack(testEdfData).reshape(len(testEdfData), -1))
        toReturn = {
            'f1_score': f1_score(testGender, y_pred),
            'auc': roc_auc_score(testGender, y_pred),
            'mcc': matthews_corrcoef(testGender, y_pred),
            'accuracy': accuracy_score(testGender, y_pred)
        }

        trainEdfTokens, validEdfTokens, testEdfTokens, trainGenders, validGenders, _testGendersCopy = get_test_train_split_from_combined()

        testEdfEnsembler = get_base_dataset(
            "combined", labels=_testGendersCopy, edfTokenPaths=testEdfTokens, is_test=True)
        label, average_pred = testEdfEnsembler.dataset.getEnsemblePrediction(
            y_pred)

        pred = np.round(average_pred)
        toReturn["ensemble_score"] = {}
        toReturn["ensemble_score"]["auc"] = roc_auc_score(label, pred)
        toReturn["ensemble_score"]["acc"] = accuracy_score(label, pred)
        toReturn["ensemble_score"]["f1_score"] = f1_score(label, pred)
        toReturn["ensemble_score"]["discordance"] = np.abs(
            pred - average_pred).mean()

        return toReturn


@ex.capture
def dl(train_split, test_split, num_epochs, lr, n_process, validation_size, max_length, use_random_ensemble, ref, num_files, use_combined, regenerate_data, model_name, use_standard_scaler, fit_generator_verbosity, steps_per_epoch):
    trainValidationDataGenerator = get_data_generator(train_split)
    if use_combined:
        trainDataGenerator = trainValidationDataGenerator
        validationDataGenerator = get_data_generator(
            train_split, split_type="validation")
    else:  # if not combined, just split on edf token level.. TODO: figure out how to use correct flow
        trainDataGenerator, validationDataGenerator = trainValidationDataGenerator.create_validation_train_split(
            validation_size=validation_size)
    model = get_model()
    print(model.summary())

    print("x batch shape", len(trainDataGenerator))
    if not regenerate_data:
        # had issues where logs get too long, so onlye one line per epoch
        # also was trying to use multiprocessing for data analysis
        if steps_per_epoch is None:
            history = model.fit_generator(trainDataGenerator, epochs=num_epochs, callbacks=get_cb_list(
            ), validation_data=validationDataGenerator, use_multiprocessing=False, workers=n_process, verbose=fit_generator_verbosity)
        else:
            history = model.fit_generator(trainDataGenerator, epochs=num_epochs, callbacks=get_cb_list(
            ), validation_data=validationDataGenerator, use_multiprocessing=False, workers=n_process, verbose=fit_generator_verbosity, steps_per_epoch=steps_per_epoch)

    testData, testGender = get_test_data()
    if use_random_ensemble:  # regenerate the dictionary structure to get correct labeling back and access to mapping back to original edfToken space
        if not use_combined:
            edfTokenPaths, testGender = cta.demux_to_tokens(
                cta.getGenderAndFileNames(test_split, ref, convert_gender_to_num=True))
            testEdfEnsembler = get_base_dataset(
                test_split, labels=testGender, edfTokenPaths=edfTokenPaths, is_test=True)
        else:
            trainEdfTokens, validEdfTokens, testEdfTokens, trainGenders, validGenders, _testGendersCopy = get_test_train_split_from_combined()
            testEdfEnsembler = get_base_dataset(
                test_split, labels=_testGendersCopy, edfTokenPaths=testEdfTokens, is_test=True)

        if use_standard_scaler:
            testGender = testEdfEnsembler.dataset.getEnsembledLabels()
        else:
            testGender = testEdfEnsembler.getEnsembledLabels()
        assert len(testData) == len(testGender)

    if regenerate_data:
        return
    model = keras.models.load_model(model_name)
    bin_acc_model = keras.models.load_model("bin_acc_" + model_name)

    # free memory so i can request less mem from 02 and get faster allocations
    del trainDataGenerator
    del validationDataGenerator

    y_pred = model.predict(testData)
    print("pred shape", y_pred.shape)
    print("test data shape", testData.shape)

    auc = roc_auc_score(testGender, y_pred.argmax(axis=1))
    f1 = f1_score(testGender, y_pred.argmax(axis=1))
    accuracy = accuracy_score(testGender, y_pred.argmax(axis=1))

    y_pred_bin_acc = bin_acc_model.predict(testData)
    print("pred shape", y_pred_bin_acc.shape)
    print("test data shape", testData.shape)

    bin_acc_auc = roc_auc_score(testGender, y_pred_bin_acc.argmax(axis=1))
    bin_acc_f1 = f1_score(testGender, y_pred_bin_acc.argmax(axis=1))
    bin_acc_accuracy = accuracy_score(
        testGender, y_pred_bin_acc.argmax(axis=1))

    toReturn = {
        'history': history.history,
        'val_scores': {
            'min_val_loss': min(history.history['val_loss']),
            'max_val_acc': max(history.history['val_binary_accuracy']),
        },
        'test_scores': {
            'f1': f1,
            'acc': accuracy,
            'auc': auc
        },
        'best_bin_acc_test_scores':  {
            'f1': bin_acc_f1,
            'acc': bin_acc_accuracy,
            'auc': bin_acc_auc
        }}
    if use_random_ensemble:
        if use_standard_scaler:
            label, average_pred = testEdfEnsembler.dataset.getEnsemblePrediction(
                y_pred.argmax(axis=1))
        else:
            label, average_pred = testEdfEnsembler.getEnsemblePrediction(
                y_pred.argmax(axis=1))
        pred = np.round(average_pred)
        toReturn["ensemble_score"] = {}
        toReturn["ensemble_score"]["auc"] = roc_auc_score(label, pred)
        toReturn["ensemble_score"]["acc"] = accuracy_score(label, pred)
        toReturn["ensemble_score"]["f1_score"] = f1_score(label, pred)
        toReturn["ensemble_score"]["discordance"] = np.abs(
            pred - average_pred).mean()
    return toReturn


if __name__ == "__main__":
    ex.run_commandline()
