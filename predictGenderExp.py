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
from sklearn.metrics import f1_score, make_scorer, accuracy_score, auc, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sacred
ex = sacred.Experiment(name="gender_predict")


ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [50, 100, 200, 400, 600],
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


@ex.config
def config():
    train_split = "train"
    test_split = "test"
    ref = "01_tcp_ar"
    parameters = {}
    clf_step = None
    clf_name = ''
    num_files = None
    freq_bins = constants.FREQ_BANDS  # bands for alpha, beta, theta, delta
    columns_to_use = constants.SMALLEST_COLUMN_SUBSET
    n_process = 7
    num_cv_folds = 5


@ex.capture
def get_data(split, ref, num_files, freq_bins, columns_to_use, n_process):
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
        split, ref, num_files=num_files, columns_to_use=columns_to_use, expand_tse=False)
    edfRawData.edf_tokens = tokenFiles[:num_files]
    edfFFTData = read.EdfFFTDatasetTransformer(
        edfRawData, n_process=n_process, freq_bins=freq_bins, return_ann=False)
    fullData = edfFFTData[:]
    # transform to number
    genders = [1 if gender == 'm' else 0 for gender in genders]
    return np.stack([datum.values.reshape(-1) for datum in fullData]), \
        np.array(genders).reshape(-1, 1)[:num_files]


@ex.capture
def getGridsearch(clf_step, parameters, n_process, num_cv_folds):
    steps = [
        clf_step
    ]
    pipeline = Pipeline(steps)
    return GridSearchCV(pipeline, parameters, cv=num_cv_folds,
                        scoring=make_scorer(f1_score), n_jobs=n_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_split, test_split, clf_name):
    trainData, trainGenders = get_data(split=train_split)
    testData, testGenders = get_data(split=train_split)
    print("Starting ", clf_name)

    gridsearch = getGridsearch()
    gridsearch.fit(trainData, trainGenders)
    print("Proportion Male in Train Set: ", trainGenders.sum() / len(trainGenders),
          "Proportion Female in Train Set: ", 1 - trainGenders.sum() / len(trainGenders))

    print("Best Parameters were: ", gridsearch.best_params_)
    print("Proportion Male in Test Set: ", testGenders.sum() / len(testGenders),
          "Proportion Female in Test Set: ", 1 - testGenders.sum() / len(testGenders))
    y_pred = gridsearch.predict(testData)
    print("Proportion Male in Predicted Test Set: ", y_pred.sum() / len(testGenders),
          "Proportion Female in Predicted Test Set: ", 1 - y_pred.sum() / len(testGenders))

    print("F1_score: ", f1_score(y_pred, testGenders))
    print("accuracy: ", accuracy_score(y_pred, testGenders))
    print("MCC: ", matthews_corrcoef(y_pred, testGenders))
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
        'mcc': matthews_corrcoef(y_pred, testGenders)
    }}


if __name__ == "__main__":
    ex.run_commandline()
