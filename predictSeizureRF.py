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
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
import pickle as pkl
import sacred
import ensembleReader as er
ex = sacred.Experiment(name="seizure_predict_traditional")



ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [50, 100, 200, 400, 600, 1200],
        'rf__max_features': ['auto', 'log2', 2, 3, 4, 5, 6, 7, 8, 12, 16],
        'rf__max_depth': [None, 4, 8, 12],
        'rf__min_samples_split': [2, 4, 8],
        'rf__n_jobs': [1],
        'rf__min_weight_fraction_leaf': [0, 0.2, 0.5],
        # 'imb__method': [None, util_funcs.ImbalancedClassResampler.SMOTE, util_funcs.ImbalancedClassResampler.RANDOM_UNDERSAMPLE]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())

@ex.named_config
def rf_debug():
    parameters = {'rf__criterion': ["gini", "entropy"],}

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
    max_samples=1000
    max_bckg_samps_per_file=20

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
    n_process = 7
    num_cv_folds = 5
    precache = True
    train_pkl="/n/scratch2/ms994/trainSeizureData.pkl"
    valid_pkl="/n/scratch2/ms994/validSeizureData.pkl"
    test_pkl="/n/scratch2/ms994/testSeizureData.pkl"
    mode = "detect"
    max_bckg_samps_per_file = 20
    resample_imbalanced_method = None
    max_samples=None
    regenerate_data=False

@ex.named_config
def predict_mode():
    mode="predict"

@ex.capture
def get_data(mode, max_samples, max_bckg_samps_per_file, ref="01_tcp_ar", num_files=None, freq_bins=[0,3.5,7.5,14,20,25,40], n_process=4, include_simple_coherence=True,):
    eds = er.EdfDatasetSegments()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()
    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file)
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file)
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file)
    train_edss = read.Flattener(read.EdfFFTDatasetTransformer(train_edss, freq_bins=freq_bins, is_pandas_data=False))[:]
    valid_edss = read.Flattener(read.EdfFFTDatasetTransformer(valid_edss, freq_bins=freq_bins, is_pandas_data=False))[:]
    test_edss = read.Flattener(read.EdfFFTDatasetTransformer(test_edss, freq_bins=freq_bins, is_pandas_data=False))[:]
    return train_edss, valid_edss, test_edss



@ex.capture
def getGridsearch(valid_indices, clf_step, parameters, n_process):
    steps = [
        # ("imb", util_funcs.ImbalancedClassResampler(n_process=n_process)),
        clf_step
    ]
    pipeline = Pipeline(steps)
    return GridSearchCV(pipeline, parameters, cv=valid_indices,
                        scoring=make_scorer(f1_score), n_jobs=n_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_pkl, valid_pkl, test_pkl, train_split, test_split, clf_name, precache, regenerate_data):
    if path.exists(train_pkl) and precache:
        testData, testLabels = pkl.load(open(test_pkl, 'rb'))
        trainData, trainLabels = pkl.load(open(train_pkl, 'rb'))
        validData, validLabels = pkl.load(open(valid_pkl, 'rb'))
    else:
        train_edss, valid_edss, test_edss = get_data()
        trainData = np.stack([datum[0] for datum in train_edss])
        trainLabels = np.stack([datum[1] for datum in train_edss])
        validData = np.stack([datum[0] for datum in valid_edss])
        validLabels = np.stack([datum[1] for datum in valid_edss])
        testData = np.stack([datum[0] for datum in test_edss])
        testLabels = np.stack([datum[1] for datum in test_edss])

        pkl.dump((trainData, trainLabels), open(train_pkl, 'wb'))
        pkl.dump((validData, validLabels), open(valid_pkl, 'wb'))
        pkl.dump((testData, testLabels), open(test_pkl, 'wb'))

    if regenerate_data:
        return

    print("Starting ", clf_name)

    trainValidData = np.vstack([trainData, validData])
    trainValidLabels = np.hstack([trainLabels, validLabels]).reshape(-1, 1)

    valid_indices = [[[i for i in range(len(trainLabels))], [i + len(trainLabels) for i in range(len(validLabels))]]]
    gridsearch = getGridsearch(valid_indices)
    gridsearch.fit(trainValidData, trainValidLabels)
    print(pd.Series(trainLabels).value_counts())

    print("Best Parameters were: ", gridsearch.best_params_)
    print(pd.Series(testLabels).value_counts())

    bestPredictor = gridsearch.best_estimator_.named_steps["rf"]
    bestPredictor.fit(trainValidData, trainValidLabels)
    y_pred = bestPredictor.predict(testData)
    print("Proportion 1 in Predicted Test Set: ", y_pred.sum() / len(testLabels),
          "Proportion 0 in Predicted Test Set: ", 1 - y_pred.sum() / len(testLabels))

    print("F1_score: ", f1_score(y_pred, testLabels))
    print("accuracy: ", accuracy_score(y_pred, testLabels))
    print("MCC: ", matthews_corrcoef(y_pred, testLabels))
    print("AUC: ", roc_auc_score(y_pred, testLabels))

    # print("auc: ", auc(y_pred, testGenders))
    toSaveDict = Dict()
    toSaveDict.getFeatureScores = getFeatureScores(gridsearch)
    toSaveDict.best_params_ = gridsearch.best_params_

    fn = "predictGender{}.pkl".format(clf_name)
    pkl.dump(toSaveDict, open(fn, 'wb'))
    ex.add_artifact(fn)

    return {'test_scores': {
        'f1': f1_score(y_pred, testLabels),
        'acc': accuracy_score(y_pred, testLabels),
        'mcc': matthews_corrcoef(y_pred, testLabels),
        'auc': roc_auc_score(y_pred, testLabels),
        "classification_report": classification_report(testLabels, y_pred, output_dict=True),
    }}


if __name__ == "__main__":
    ex.run_commandline()
