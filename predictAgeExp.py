import sacred
ex = sacred.Experiment(name="age_learning_exp")

#Sanity check to see if we can do something fundamental like this

import data_reader as read
import util_funcs
import pandas as pd
import numpy as np
from os import path
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from addict import Dict
import pickle as pkl

from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

@ex.named_config
def simple_pca_lr_config():
    use_simple_lr_pca_pipeline = True
    hyperparameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2],
        'lr__solver': ["sag"],
        'lr__max_iter': [250],
        'lr__n_jobs': [1]
    }

@ex.config
def use_lstm():
    use_simple_lstm = True


@ex.config
def config():
    ref = "01_tcp_ar"
    train_split = "train"
    test_split = "dev_test"
    n_process = 6
    precache = True
    num_files = None
    use_simple_lr_pca_pipeline = False
    use_simple_lstm = False
    hyperparameters = None
    window = None
    non_overlapping = True
    kbins_strat = "quantile"
    validation_size = 0.2
    prestore_data = False
    precached_pkl = None



@ex.capture
def get_data(split, ref, n_process, precache, num_files, window, non_overlapping):
    ageData = read.getAgesAndFileNames(split, ref)
    if num_files is not None:
        ageData = ageData[0:num_files]
    clinical_txt_paths = [ageDatum[0] for ageDatum in ageData]
    ages = [ageDatum[1] for ageDatum in ageData]

    #associate first token file with each session for now
    tokenFiles = []
    for session_file in clinical_txt_paths:
        session_dir = path.dirname(session_file)
        session_tkn_files = read.get_token_file_names(session_dir)
        session_tkn_files.sort()
        tokenFiles.append(session_tkn_files[0])
    edfReader = read.EdfDataset(split, ref, expand_tse=False) #discarding the annotations eventually
    edfReader.edf_tokens = tokenFiles #override to use only eegs with ages we have
    edfFFTData = read.EdfFFTDatasetTransformer(edf_dataset=edfReader, n_process=n_process, precache=True, window_size=window, non_overlapping=non_overlapping, return_ann=False)
    return edfFFTData[:], ages, clinical_txt_paths


@ex.main
def main(use_simple_lr_pca_pipeline, kbins_strat, train_split, test_split, hyperparameters, validation_size, n_process, precached_pkl, prestore_data):
    if precached_pkl is not None:
        allData = pkl.load(precached_pkl)
        data = precached_pkl["data"]
        ages = precached_pkl["ages"]
        # clinical_txt_paths = precached_pkl["clinical_txt_paths"]
        testAges = precached_pkl["testAges"]
        testData = precached_pkl["testData"]
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
        pkl.dump(toStore, open("agePredictionData.pkl", 'wb'))
        return


    if use_simple_lr_pca_pipeline:
        kbins = KBinsDiscretizer(10, encode='ordinal', strategy=kbins_strat)
        ages = np.array(ages).reshape(-1, 1)
        ages = kbins.fit_transform(ages)
        return_dict['kbins'] = kbins.bin_edges_
        testAges = np.array(testAges).reshape(-1, 1)
        testAges = kbins.transform(testAges)
        data = np.stack(data).reshape(len(data), -1)
        testData = np.stack(testData).reshape(len(data), -1)

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

        y_pred = best_pipeline.predict(data)
        test_score = f1_score(testAges, y_pred, average="weighted")
        print("test_score: {}".format(test_score))
        return_dict["test_score"] = test_score
        return return_dict

    raise Exception()
    print("hi")

if __name__ == "__main__":
    ex.run_commandline()
