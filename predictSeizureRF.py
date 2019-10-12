import tsfresh.feature_extraction.feature_calculators as tsf
from sacred.observers import MongoObserver
from sacred import SETTINGS
SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False #weird issue where GridSearchCV alters one of the config values
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
from os import path
import sys
import data_reader as read
import constants
import util_funcs
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
import pickle as pkl
import sacred
import ensembleReader as er
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
ex = sacred.Experiment(name="seizure_predict_traditional_ml")



# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [50, 100, 200, 400, 600, 1200],
        'rf__max_features': ['auto', 'log2', 2, 3, 4, 5, 6, 7, 8, 12, 16],
        'rf__max_depth': [None, 4, 8, 12], #smaller max depth, gradient boosting, more max features
        'rf__min_samples_split': [2, 4, 8],
        'rf__n_jobs': [1],
        'rf__min_weight_fraction_leaf': [0, 0.2, 0.5],
        # 'imb__method': [None, util_funcs.ImbalancedClassResampler.SMOTE, util_funcs.ImbalancedClassResampler.RANDOM_UNDERSAMPLE]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())

@ex.named_config
def xgboost():
    parameters = {
        "xgboost__max_depth": [3,4,5,6],
        "xgboost__learning_rate":[0.1,0.2],
        "xgboost__gamma":[0,0.1,0.2],
        "xgboost__alpha":[0,0.1,0.2],
        "xgboost__lambda":[0,0.1,0.2],
        "xgboost__top_k":[0,1,2,4,8,16],
        "xgboost__feature_selector":["cyclic", "shuffle"],
        "xgboost__rf__n_estimators":[100,200,300,400,600],
    }
    clf_name = "xgboost"
    clf_step = (clf_name, xgb.XGBRegressor(objective='binary:logistic',))
    use_xgboost=True

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

@ex.named_config
def knn_server():
    train_pkl="/home/msaqib/trainSeizureData.pkl"
    valid_pkl="/home/msaqib/validSeizureData.pkl"
    test_pkl="/home/msaqib/testSeizureData.pkl"

@ex.config
def config():
    train_split = "train"
    use_random_cv = False
    test_split = "dev_test"
    ref = "01_tcp_ar"
    include_simple_coherence = True
    parameters = {}
    clf_step = None
    clf_name = ''
    num_files = None
    freq_bins=[0,3.5,7.5,14,20,25,40]
    n_process = 7
    precache = True
    train_pkl="/n/scratch2/ms994/trainSeizureData.pkl"
    valid_pkl="/n/scratch2/ms994/validSeizureData.pkl"
    test_pkl="/n/scratch2/ms994/testSeizureData.pkl"
    mode = er.EdfDatasetSegmentedSampler.DETECT_MODE
    max_bckg_samps_per_file = 30
    resample_imbalanced_method = None
    max_samples=None
    regenerate_data=False
    imbalanced_resampler = None
    pre_cooldown=4
    post_cooldown=None
    sample_time=4
    num_seconds=1
    mode=er.EdfDatasetSegmentedSampler.DETECT_MODE
    use_xgboost = False
    use_simple_hand_engineered_features=False


@ex.named_config
def predict_mode_knn_server():
    mode=er.EdfDatasetSegmentedSampler.PREDICT_MODE
    train_pkl="/home/msaqib/trainPredictSeizureData.pkl"
    valid_pkl="/home/msaqib/validPredictSeizureData.pkl"
    test_pkl="/home/msaqib/testPredictSeizureData.pkl"

@ex.named_config
def predict_mode():
    mode=er.EdfDatasetSegmentedSampler.PREDICT_MODE
    train_pkl="/n/scratch2/ms994/trainPredictSeizureData.pkl"
    valid_pkl="/n/scratch2/ms994/validPredictSeizureData.pkl"
    test_pkl="/n/scratch2/ms994/testPredictSeizureData.pkl"

@ex.capture
def getDataSampleGenerator(pre_cooldown, post_cooldown, sample_time, num_seconds, n_process):
    return er.EdfDatasetSegments(pre_cooldown=pre_cooldown, post_cooldown=post_cooldown, sample_time=sample_time, num_seconds=num_seconds, n_process=n_process)


@ex.capture
def get_data(mode, max_samples, n_process, max_bckg_samps_per_file,use_simple_hand_engineered_features, ref="01_tcp_ar", num_files=None, freq_bins=[0,3.5,7.5,14,20,25,40],  include_simple_coherence=True,):
    eds = getDataSampleGenerator()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()

    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=n_process, )[:]
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=n_process, )[:]
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, mode=mode, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=n_process, )[:]

    def simple_edss(edss):
        '''
        Use only a few columns so that we don't make 21*20 coherence pairs
        '''
        all_channels = util_funcs.get_common_channel_names()
        subset_channels = constants.SYMMETRIC_COLUMN_SUBSET
        subset_channels = [all_channels.index(channel) for channel in subset_channels]
        return [(datum[0][:, subset_channels], datum[1]) for datum in edss]
    if include_simple_coherence:
        trainCoherData = np.stack([datum.values for datum in [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(train_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
        validCoherData = np.stack([datum.values for datum in [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(valid_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
        testCoherData = np.stack([datum.values for datum in  [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(test_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
    if use_simple_hand_engineered_features:
        trainSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(train_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]
        validSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(valid_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]
        testSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(test_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]

    train_edss = read.Flattener(read.EdfFFTDatasetTransformer(train_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    valid_edss = read.Flattener(read.EdfFFTDatasetTransformer(valid_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    test_edss = read.Flattener(read.EdfFFTDatasetTransformer(test_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    def split_tuples(data):
        return np.stack([datum[0] for datum in data]), np.stack([datum[1] for datum in data])
    train_edss, train_labels = split_tuples(train_edss)
    valid_edss, valid_labels = split_tuples(valid_edss)
    test_edss, test_labels = split_tuples(test_edss)


    if include_simple_coherence:
        train_edss = np.hstack([train_edss, trainCoherData])
        valid_edss = np.hstack([valid_edss, validCoherData])
        test_edss = np.hstack([test_edss, testCoherData])

    if use_simple_hand_engineered_features:
        train_edss = np.hstack([train_edss, np.stack(trainSHED)])
        valid_edss = np.hstack([valid_edss, np.stack(validSHED)])
        test_edss = np.hstack([test_edss, np.stack(testSHED)])


    return (train_edss, train_labels), (valid_edss, valid_labels), (test_edss, test_labels)

@ex.capture
def getImbResampler(imbalanced_resampler):
    if imbalanced_resampler is None:
        return None
    elif imbalanced_resampler == "SMOTE":
        return SMOTE()
    elif imbalanced_resampler == "rul":
        return RandomUnderSampler()

@ex.capture
def resample_x_y(x, y, imbalanced_resampler):
    if imbalanced_resampler is None:
        return x, y
    else:
        return getImbResampler().fit_resample(x, y)

@ex.capture
def getGridsearch(valid_indices, clf_step, parameters, n_process, use_random_cv, num_random_choices=10, use_xgboost=False):
    steps = [
        # ("imb", util_funcs.ImbalancedClassResampler(n_process=n_process)),
        clf_step
    ]



    pipeline = Pipeline(steps)
    if use_xgboost:
        scorer = make_scorer(roc_auc_score)
    else:
        scorer = make_scorer(f1_score)
    if use_random_cv:
        return RandomizedSearchCV(pipeline, Dict(parameters), cv=valid_indices,
                            scoring=scorer, n_jobs=n_process, n_iter=num_random_choices)
    return GridSearchCV(pipeline, Dict(parameters), cv=valid_indices,
                        scoring=scorer, n_jobs=n_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_pkl, valid_pkl, test_pkl, train_split, mode, imbalanced_resampler, test_split, clf_name, precache, regenerate_data, use_xgboost, num_random_choices=10):
    if path.exists(train_pkl) and precache:
        testData, testLabels = pkl.load(open(test_pkl, 'rb'))
        trainData, trainLabels = pkl.load(open(train_pkl, 'rb'))
        validData, validLabels = pkl.load(open(valid_pkl, 'rb'))
    else:
        train_edss, valid_edss, test_edss = get_data()
        trainData, trainLabels = train_edss
        validData, validLabels = valid_edss
        testData, testLabels = test_edss


        pkl.dump((trainData, trainLabels), open(train_pkl, 'wb'))
        pkl.dump((validData, validLabels), open(valid_pkl, 'wb'))
        pkl.dump((testData, testLabels), open(test_pkl, 'wb'))

    if regenerate_data:
        return

    print("Starting ", clf_name)

    #resample separately to avoid any data leaking between splits
    trainDataResampled, trainLabelsResampled = resample_x_y(trainData, trainLabels)
    validDataResampled, validLabelsResampled = resample_x_y(validData, validLabels)


    # if use_xgboost:
    #     return xgboost_flow(trainDataResampled, trainLabelsResampled, validDataResampled, validLabelsResampled, testData, testLabels)

    trainValidData = np.vstack([trainDataResampled, validDataResampled])
    trainValidLabels = np.hstack([trainLabelsResampled, validLabelsResampled])


    valid_indices = [[[i for i in range(len(trainLabelsResampled))], [i + len(trainLabelsResampled) for i in range(len(validLabelsResampled))]]]
    gridsearch = getGridsearch(valid_indices)

    gridsearch.fit(trainValidData, trainValidLabels)
    print(pd.Series(trainLabels).value_counts())

    print("Best Parameters were: ", gridsearch.best_params_)
    print(pd.Series(testLabels).value_counts())

    bestPredictor = gridsearch.best_estimator_.named_steps[clf_name]
    bestPredictor.fit(trainValidData, trainValidLabels)
    y_pred = bestPredictor.predict(testData)

    if y_pred.dtype == np.float32 or y_pred.dtype == np.float:
        y_pred = y_pred > 0.5

    print("Proportion True in Predicted Test Set: ", y_pred.sum() / len(testLabels),
          "Proportion False in Predicted Test Set: ", 1 - y_pred.sum() / len(testLabels))

    print("F1_score: ", f1_score(y_pred, testLabels))
    print("accuracy: ", accuracy_score(y_pred, testLabels))
    print("MCC: ", matthews_corrcoef(y_pred, testLabels))
    print("AUC: ", roc_auc_score(y_pred, testLabels))

    # print("auc: ", auc(y_pred, testGenders))
    toSaveDict = Dict()
    toSaveDict.getFeatureScores = getFeatureScores(gridsearch)
    toSaveDict.gridsearch = gridsearch
    toSaveDict.best_params_ = gridsearch.best_params_

    fn = "predictGender{}_{}_{}.pkl".format(clf_name, mode, imbalanced_resampler if imbalanced_resampler is not None else "noresample")
    pkl.dump(toSaveDict, open(fn, 'wb'))
    # ex.add_artifact(fn)

    return {'test_scores': {
        'f1': f1_score(y_pred, testLabels),
        'acc': accuracy_score(y_pred, testLabels),
        'mcc': matthews_corrcoef(y_pred, testLabels),
        'auc': roc_auc_score(y_pred, testLabels),
        "classification_report": classification_report(testLabels, y_pred, output_dict=True),
        "best_params":  gridsearch.best_params_
    }}


if __name__ == "__main__":
    #https://github.com/ContinuumIO/anaconda-issues/issues/11294#issuecomment-533138984 mp got jacked by version upgrade

    ex.run_commandline()
