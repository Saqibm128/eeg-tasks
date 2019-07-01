import sacred
ex = sacred.Experiment(name="age_learning_exp")

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Activation, Dropout
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences

#Sanity check to see if we can do something fundamental like this

import data_reader as read
import util_funcs
import pandas as pd
import numpy as np
from os import path
from sklearn.metrics import f1_score, make_scorer, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from addict import Dict
import pickle as pkl

from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

@ex.named_config
def simple_pca_lin_reg_config():
    use_simple_lin_reg_pca_pipeline = True

@ex.named_config
def simple_pca_lr_config():
    use_simple_lr_pca_pipeline = True
    discretize_age = True
    hyperparameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2],
        'lr__solver': ["sag"],
        'lr__max_iter': [250],
        'lr__n_jobs': [1]
    }

@ex.named_config
def use_lstm():
    discretize_age = True
    use_simple_lstm = True
    kbins_encoding = "onehot-dense"
    input_shape = (None, None, len(read.EdfFFTDatasetTransformer.freq_bins)) #variable batch, variable time steps, but constant num features

@ex.named_config
def bpm():
    return_mode="bpm"


@ex.config
def config():
    input_shape = ()
    ref = "01_tcp_ar"
    train_split = "train"
    test_split = "dev_test"
    n_process = 6
    k_bins = 10
    precache = True
    num_epochs = 1000
    num_files = None
    use_simple_lr_pca_pipeline = False
    use_simple_lin_reg_pca_pipeline = False
    use_simple_lstm = False
    hyperparameters = {}
    window = None
    non_overlapping = True
    kbins_strat = "quantile"
    kbins_encoding = "ordinal"
    validation_size = 0.2
    prestore_data = False
    precached_pkl = None
    return_mode = "age"
    input_shape = ()
    latent_shape = (200)
    filter = True
    discretize_age = False
    lr = 0.001

@ex.capture
def get_lstm(input_shape, latent_shape, k_bins, lr):
    model = Sequential([
        LSTM(latent_shape),
        Dense(units=latent_shape),
        Dropout(0.5),
        Activation("relu"),
        Dense(units=k_bins),
    ])
    sgd = optimizers.SGD(lr=lr, clipnorm=1.)
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model




@ex.capture
def get_data(split, ref, n_process, precache, num_files, window, non_overlapping, return_mode, filter):
    if return_mode=="age":
        ageData = read.getAgesAndFileNames(split, ref)
    elif return_mode=="bpm":
        ageData = read.getBPMAndFileNames(split, ref) #not really agedata, bpmdata really
    if num_files is not None:
        ageData = ageData[0:num_files]
    clinical_txt_paths = [ageDatum[0] for ageDatum in ageData]
    ages = [ageDatum[1] for ageDatum in ageData]

    #associate first token file with each session for now
    tokenFiles = []
    problemFiles = []
    for session_file in clinical_txt_paths:
        session_dir = path.dirname(session_file)
        session_tkn_files = read.get_token_file_names(session_dir)
        session_tkn_files.sort()
        tokenFiles.append(session_tkn_files[0])
    edfReader = read.EdfDataset(split, ref, expand_tse=False, filter=filter) #discarding the annotations eventually
    edfReader.edf_tokens = tokenFiles #override to use only eegs with ages we have
    edfFFTData = read.EdfFFTDatasetTransformer(edf_dataset=edfReader, n_process=n_process, precache=True, window_size=window, non_overlapping=non_overlapping, return_ann=False)
    return edfFFTData[:], ages, clinical_txt_paths


@ex.main
def main(
    use_simple_lr_pca_pipeline,
    kbins_strat,
    train_split,
    test_split,
    hyperparameters,
    k_bins,
    validation_size,
    n_process,
    precached_pkl,
    prestore_data,
    return_mode,
    use_simple_lin_reg_pca_pipeline,
    use_simple_lstm,
    discretize_age,
    kbins_encoding,
    num_epochs
    ):
    if precached_pkl is not None:
        allData = pkl.load(open(precached_pkl, 'rb'))
        data = allData["data"]
        # clinical_txt_paths = precached_pkl["clinical_txt_paths"]
        ages = allData["ages"]
        testAges = allData["testAges"]
        testData = allData["testData"]
        # test_clinical_txt_paths = precached_pkl["test_clinical_txt_paths"]
    else:
        data, ages, clinical_txt_paths = get_data(split=train_split)
        testData, testAges, test_clinical_txt_paths = get_data(split=test_split)
    return_dict = Dict()

    if prestore_data:
        toStore = Dict()
        toStore.data = data
        toStore.ages = ages
        toStore.clinical_txt_paths = clinical_txt_paths
        toStore.testData = testData
        toStore.testAges = testAges
        toStore.test_clinical_txt_paths = test_clinical_txt_paths
        if return_mode == "age":
            pkl.dump(toStore, open("agePredictionData.pkl", 'wb'))
        elif return_mode == "bpm":
            pkl.dump(toStore, open("bpmPredictionData.pkl", 'wb'))
        return return_mode

    if discretize_age:
        kbins = KBinsDiscretizer(k_bins, encode=kbins_encoding, strategy=kbins_strat)
        ages = np.array(ages).reshape(-1, 1)
        ages = kbins.fit_transform(ages)
        return_dict['kbins'] = kbins.bin_edges_
        testAges = np.array(testAges).reshape(-1, 1)
        testAges = kbins.transform(testAges)

    if use_simple_lstm:
        model = get_lstm()
        x = pad_sequences(data)
        model.fit(x, ages, epochs=num_epochs, validation_split=validation_size)
        testX = pad_sequences(testData)
        score = model.evaluate(testX, testAges)
        model.save("model.h5")
        ex.add_artifact("model.h5")
        return score

    if use_simple_lin_reg_pca_pipeline:
        ages = np.array(ages).reshape(-1, 1)
        testAges = np.array(testAges).reshape(-1, 1)
        data = np.stack(data).reshape(len(data), -1)
        testData = np.stack(testData).reshape(len(testData), -1)
        steps = [
            ('pca', PCA(n_components=10)),
            ('scaler', StandardScaler()),
            ('lin_reg', LinearRegression()),
        ]
        p = Pipeline(steps)
        cv = int(1/validation_size)
        gridsearch = GridSearchCV(p, hyperparameters, scoring=make_scorer(mean_squared_error), cv=cv, n_jobs=n_process)
        gridsearch.fit(data, ages)
        return_dict["gridsearch_best_estimator"] = gridsearch.best_estimator_
        return_dict["best_cv_score"] = gridsearch.best_score_
        print("best cv score was {}".format(gridsearch.best_score_))
        best_pipeline = gridsearch.best_estimator_
        best_pipeline.fit(data, ages)

        y_pred = best_pipeline.predict(testData)
        test_score = mean_squared_error(testAges, y_pred)
        print("test_score: {}".format(test_score))
        return_dict["test_score"] = test_score
        pkl.dump(return_dict, open("predict_{}Exp.pkl".format(return_mode), 'wb'))
        ex.add_artifact("predict_{}Exp.pkl".format(return_mode))
        return test_score


    if use_simple_lr_pca_pipeline:
        data = np.stack(data).reshape(len(data), -1)
        testData = np.stack(testData).reshape(len(testData), -1)

        steps = [
            ('pca', PCA(n_components=10)),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression()),
        ]
        p = Pipeline(steps)
        cv = int(1/validation_size)
        gridsearch = GridSearchCV(p, hyperparameters, scoring=make_scorer(f1_score, average="weighted"), cv=cv, n_jobs=n_process)
        gridsearch.fit(data, ages)
        return_dict["gridsearch_best_estimator"] = gridsearch.best_estimator_
        return_dict["best_cv_score"] = gridsearch.best_score_
        print("best cv score was {}".format(gridsearch.best_score_))
        best_pipeline = gridsearch.best_estimator_
        best_pipeline.fit(data, ages)

        y_pred = best_pipeline.predict(testData)
        test_score = f1_score(testAges, y_pred, average="weighted")
        print("test_score: {}".format(test_score))
        return_dict["test_score"] = test_score
        pkl.dump(return_dict, open("predict_{}Exp.pkl".format(return_mode), 'wb'))
        ex.add_artifact("predict_{}Exp.pkl".format(return_mode))
        return test_score

    raise Exception("Valid config not set")

if __name__ == "__main__":
    ex.run_commandline()
