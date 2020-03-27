import sys, os
sys.path.append(os.path.realpath(".."))
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
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
import pickle as pkl
import sacred
import ensembleReader as er
ex = sacred.Experiment(name="gender_predict_conventional")

'''
Based on
https://www.nature.com/articles/s41598-018-21495-7
https://www.sciencedirect.com/science/article/pii/S0028393210004100?via%3Dihub
'''


ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [50, 100, 200, 400, 600, 1200],
        'rf__max_features': ['auto', 'log2'], #, 2, 3, 4, 5, 6, 7, 8, 16, .04, .02
        'rf__max_depth': [None, 4, 8, 12],
        'rf__min_samples_split': [2, 4, 8],
        'rf__n_jobs': [1],
        'rf__min_weight_fraction_leaf': [0, 0.2, 0.5]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())

@ex.named_config
def linked_ear():
    ref = "02_tcp_le"
    train_pkl="trainGenderDataLe.pkl"
    test_pkl="testGenderDataLe.pkl"


@ex.named_config
def lr():
    parameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2, .4, .8],
        'lr__solver': ["sag"],
        'lr__max_iter': [1000],
        'lr__n_jobs': [1]
    }
    clf_name = "lr"
    clf_step = ('lr', LogisticRegression())


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
    freq_bins = constants.FREQ_BANDS  # bands for alpha, beta, theta, delta
    columns_to_use = constants.SYMMETRIC_COLUMN_SUBSET
    n_process = 7
    num_cv_folds = 5
    n_gridsearch_process = n_process
    precache = False
    train_pkl="trainGenderData.pkl"
    filter = True
    test_pkl="testGenderData.pkl"

@ex.capture
def get_data(split, ref="01_tcp_ar", num_files=None, freq_bins=[0,3.5,7.5,14,20,25,40], columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=4, include_simple_coherence=True, filter=True):
    genderDictItems = cta.getGenderAndFileNames(split, ref)
    clinicalTxtPaths = [genderDictItem[0]
                        for genderDictItem in genderDictItems]
    singGenders = [genderDictItem[1] for genderDictItem in genderDictItems]
    tokenFiles = []
    genders = []  # duplicate singGenders depending on number of tokens per session
    for i, txtPath in enumerate(clinicalTxtPaths):
        session_dir = path.dirname(txtPath)
        session_tkn_files = sorted(read.get_token_file_names(session_dir))
        tokenFiles += session_tkn_files
        genders += [singGenders[i] for tkn_file in session_tkn_files]
    edfRawData = read.EdfDataset(
        split, ref, num_files=num_files, columns_to_use=columns_to_use, expand_tse=False, filter=filter)
    edfRawData.edf_tokens = tokenFiles[:num_files]
    edfFFTData = read.EdfFFTDatasetTransformer(
        edfRawData, n_process=n_process, freq_bins=freq_bins, return_ann=False)
    fullData = edfFFTData[:]
    # transform to number
    genders = [1 if gender == 'm' else 0 for gender in genders]

    toReturnData = np.stack([datum.values.reshape(-1) for datum in fullData])

    if include_simple_coherence:
        coherData = wfdata.CoherenceTransformer(edfRawData, columns_to_use=columns_to_use, n_process=n_process)
        fullCoherData = [datum[0] for datum in coherData[:]]
        fullCoherData = np.stack([datum.values for datum in fullCoherData])
        toReturnData = np.hstack([toReturnData, fullCoherData])


    return toReturnData, \
        np.array(genders).reshape(-1, 1)[:num_files]


@ex.capture
def getGridsearch(clf_step, parameters, n_gridsearch_process, num_cv_folds):
    steps = [
        clf_step
    ]
    pipeline = Pipeline(steps)
    return GridSearchCV(pipeline, parameters, cv=num_cv_folds,
                        scoring=make_scorer(f1_score), n_jobs=n_gridsearch_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_pkl, test_pkl, train_split, test_split, clf_name, precache):
    if path.exists(train_pkl) and precache:
        trainData, trainGenders = pkl.load(open(train_pkl, 'rb'))
        # ex.add_resource(train_pkl) #pushes gbs of data to mongo
    else:
        trainData, trainGenders = get_data(split=train_split)
        pkl.dump((trainData, trainGenders), open(train_pkl, 'wb'))
        # ex.add_artifact(train_pkl) #pushes gbs of data to mongo

    if path.exists(test_pkl) and precache:
        testData, testGenders = pkl.load(open(test_pkl, 'rb'))
        # ex.add_resource(train_pkl) #pushes gbs of data to mongo
    else:
        testData, testGenders = get_data(split=test_split)
        pkl.dump((testData, testGenders), open(test_pkl, 'wb'))
        # ex.add_artifact(test_pkl) #pushes gbs of data to mongo
    print("Starting ", clf_name)

    gridsearch = getGridsearch()
    gridsearch.fit(trainData, trainGenders)
    print("Proportion Male in Train Set: ", trainGenders.sum() / len(trainGenders),
          "Proportion Female in Train Set: ", 1 - trainGenders.sum() / len(trainGenders))

    print("Best Parameters were: ", gridsearch.best_params_)
    print("Proportion Male in Test Set: ", testGenders.sum() / len(testGenders),
          "Proportion Female in Test Set: ", 1 - testGenders.sum() / len(testGenders))
    bestPredictor = gridsearch.best_estimator_
    bestPredictor.fit(trainData, trainGenders)
    y_pred = bestPredictor.predict(testData)
    print("Proportion Male in Predicted Test Set: ", y_pred.sum() / len(testGenders),
          "Proportion Female in Predicted Test Set: ", 1 - y_pred.sum() / len(testGenders))

    print("F1_score: ", f1_score(y_pred, testGenders))
    print("accuracy: ", accuracy_score(y_pred, testGenders))
    print("MCC: ", matthews_corrcoef(y_pred, testGenders))
    print("AUC: ", roc_auc_score(y_pred, testGenders))

    # print("auc: ", auc(y_pred, testGenders))
    toSaveDict = Dict()
    toSaveDict.getFeatureScores = getFeatureScores(gridsearch)
    toSaveDict.best_params_ = gridsearch.best_params_

    fn = "predictGender{}.pkl".format(clf_name)
    pkl.dump(toSaveDict, open(fn, 'wb'))
    ex.add_artifact(fn)

    return {'test_scores': {
        'f1': f1_score(y_pred, testGenders),
        'acc': accuracy_score(y_pred, testGenders),
        'mcc': matthews_corrcoef(y_pred, testGenders),
        'auc': roc_auc_score(y_pred, testGenders),

    }, 'gridsearch_results': {
        'best_params_': toSaveDict.best_params_,
        'cv_results': gridsearch.cv_results_
    }}


if __name__ == "__main__":
    ex.run_commandline()
