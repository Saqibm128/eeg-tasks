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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import wf_analysis.datasets as wfdata
import pickle as pkl
import sacred
ex = sacred.Experiment(name="heartrate_predict")

'''
Based on
https://www.nature.com/articles/s41598-018-21495-7
https://www.sciencedirect.com/science/article/pii/S0028393210004100?via%3Dihub
'''


# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [50, 100, 200, 400, 600, 1000],
        'rf__max_features': ['auto', 'log2', .1, .4, .6, .8],
        'rf__max_depth': [None, 2, 4, 6, 8, 10, 12],
        'rf__min_samples_split': [2, 4, 8],
        'rf__n_jobs': [1],
        'rf__min_weight_fraction_leaf': [0, 0.2, 0.5]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())


@ex.named_config
def lr():
    parameters = {
        'lr__n_jobs': [1]
    }
    clf_name = "lr"
    clf_step = ('lr', LinearRegression())


@ex.named_config
def debug():
    num_files = 200


@ex.named_config
def use_all_columns():
    columns_to_use = util_funcs.get_common_channel_names()

@ex.named_config
def use_single_hemispheres():
    columns_to_use = constants.SMALLEST_COLUMN_SUBSET

@ex.config
def config():
    train_split = "train"
    test_split = "dev_test"
    ref = "01_tcp_ar"
    include_simple_coherence = True
    parameters = {}
    clf_step = None
    clf_name = ''
    num_files = None
    freq_bins = None # if set to None, use KBins to find quantile bins to categorize bpms into, then just calculaate frequencies from that
    columns_to_use = constants.SYMMETRIC_COLUMN_SUBSET
    n_process = 7
    num_cv_folds = 5
    n_gridsearch_process = n_process
    precache = False
    train_pkl="trainBPMData.pkl"
    test_pkl="testBPMData.pkl"
    num_kbins = 10

@ex.capture
def get_freq_bins(freq_bins,  num_kbins, bpms=None,):
    if freq_bins is not None:
        return freq_bins
    if bpms is not None:
        kbins = KBinsDiscretizer(num_kbins)
        kbins.fit(np.array(bpms).reshape(-1,1))
        edges = kbins.bin_edges_[0]
        freqs = [edge / 60 for edge in edges]
        return freqs

@ex.capture
def get_data(split, ref, num_files, freq_bins, columns_to_use, n_process, include_simple_coherence):
    BPMDictItems = cta.getBPMAndFileNames(split, ref)
    clinicalTxtPaths = [BPMDictItem[0]
                        for BPMDictItem in BPMDictItems]
    singBPMs = [BPMDictItem[1] for BPMDictItem in BPMDictItems]
    tokenFiles = []
    BPMs = []  # duplicate singBPMs depending on number of tokens per session
    for i, txtPath in enumerate(clinicalTxtPaths):
        session_dir = path.dirname(txtPath)
        session_tkn_files = sorted(read.get_token_file_names(session_dir))
        tokenFiles += session_tkn_files
        BPMs += [singBPMs[i] for tkn_file in session_tkn_files]
    edfRawData = read.EdfDataset(
        split, ref, num_files=num_files, columns_to_use=columns_to_use, expand_tse=False)
    edfRawData.edf_tokens = tokenFiles[:num_files]

    #figure out freq_bins if none was passed in based on BPMs (should only happen for train set)
    freq_bins = get_freq_bins(freq_bins, bpms=BPMs[:num_files])

    edfFFTData = read.EdfFFTDatasetTransformer(
        edfRawData, n_process=n_process, freq_bins=freq_bins, return_ann=False)
    fullData = edfFFTData[:]

    toReturnData = np.stack([datum.values.reshape(-1) for datum in fullData])

    if include_simple_coherence:
        coherData = wfdata.CoherenceTransformer(edfRawData, columns_to_use=columns_to_use, n_process=n_process)
        fullCoherData = [datum[0] for datum in coherData[:]]
        fullCoherData = np.stack([datum.values for datum in fullCoherData])
        toReturnData = np.hstack([toReturnData, fullCoherData])


    return toReturnData, \
        np.array(BPMs).reshape(-1, 1)[:num_files]


@ex.capture
def getGridsearch(clf_step, parameters, n_gridsearch_process, num_cv_folds):
    steps = [
        clf_step
    ]
    pipeline = Pipeline(steps)
    return GridSearchCV(pipeline, parameters, cv=num_cv_folds,
                        scoring=make_scorer(r2_score), n_jobs=n_gridsearch_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_pkl, test_pkl, train_split, test_split, clf_name, precache, freq_bins):
    if path.exists(train_pkl) and precache:
        trainData, trainBPMs = pkl.load(open(train_pkl, 'rb'))
        ex.add_resource(train_pkl)
    else:
        trainData, trainBPMs = get_data(split=train_split)
        pkl.dump((trainData, trainBPMs), open(train_pkl, 'wb'))
        ex.add_artifact(train_pkl)

    freq_bins = get_freq_bins(freq_bins, bpms=trainBPMs)
    if path.exists(test_pkl) and precache:
        testData, testBPMs = pkl.load(open(train_pkl, 'rb'))
        ex.add_resource(train_pkl)
    else:
        testData, testBPMs = get_data(split=test_split, freq_bins=freq_bins)
        pkl.dump((testData, testBPMs), open(test_pkl, 'wb'))
        ex.add_artifact(test_pkl)
    print("Starting ", clf_name)

    gridsearch = getGridsearch()
    gridsearch.fit(trainData, trainBPMs)
    print("Best Parameters were: ", gridsearch.best_params_)
    bestPredictor = gridsearch.best_estimator_
    bestPredictor.fit(trainData, trainBPMs)
    y_pred = bestPredictor.predict(testData)
    print("r2_score: ", r2_score(y_pred, testBPMs))

    # print("auc: ", auc(y_pred, testBPMs))
    toSaveDict = Dict()
    toSaveDict.getFeatureScores = getFeatureScores(gridsearch)
    toSaveDict.best_params_ = gridsearch.best_params_

    fn = "predictBPM{}.pkl".format(clf_name)
    pkl.dump(toSaveDict, open(fn, 'wb'))
    ex.add_artifact(fn)

    return {'test_scores': {
        'r2_score': r2_score(y_pred, testBPMs),
    }}


if __name__ == "__main__":
    ex.run_commandline()
