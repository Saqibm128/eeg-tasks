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
from wf_analysis.spatialTemporalDatasets import BasicSpatialDataset
from keras_models.dataGen import EdfDataGenerator
from keras_models.vanPutten import vp_conv2d
from keras import optimizers
import pickle as pkl
import sacred
ex = sacred.Experiment(name="gender_predict_conv")

from sacred.observers import MongoObserver
# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def debug():
    max_length=constants.COMMON_FREQ
    num_files=20
    batch_size=8
    num_epochs=1

@ex.config
def config():
    train_split = "train"
    test_split = "dev_test"
    ref = "01_tcp_ar"
    n_process = 7
    num_files = None
    max_length = 2 * constants.COMMON_FREQ
    batch_size = 32
    dropout = 0.25
    spatialMapping = constants.SIMPLE_CONV2D_MAP
    use_early_stopping = True
    patience = 10
    num_epochs = 100
    lr = 0.0001



@ex.capture
def get_data(split, ref, n_process, num_files, max_length):
    genderDict = cta.getGenderAndFileNames(split, ref)
    edfTokenPaths, genders = cta.demux_to_tokens(genderDict)
    edfData = read.EdfDataset(split, ref, n_process=n_process, max_length=max_length * pd.Timedelta(seconds=constants.COMMON_DELTA))
    edfData.edf_tokens = edfTokenPaths[:num_files]
    genders = [1 if item[1]=='m' else 0 for item in genderDict][:num_files]
    return edfData, genders

@ex.capture
def get_simple_mapping_data(split, batch_size, spatialMapping, num_files, max_length, num_epochs):
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
    edfData, genders = get_data(split)
    edfData = edfData[:]
    edfData.use_mp = False
    return EdfDataGenerator(edfData, n_classes=2, labels=np.array(genders), batch_size=batch_size, max_length=max_length)

@ex.capture
def get_model(dropout, spatialMapping, max_length,lr):
    model = (dropout=dropout, input_shape=(max_length + 1, len(spatialMapping), len(spatialMapping[0]), 1))
    adam = optimizers.Adam(lr=lr)
    model.compile(adam, loss="categorical_crossentropy", metrics=["binary_accuracy"])
    return model

@ex.main
def main(train_split, test_split, spatialMapping, num_epochs, lr, n_process):
    trainDataGenerator = get_simple_mapping_data(train_split)
    model = get_model()
    x, y  = trainDataGenerator[0]
    y_pred = model.predict(x)
    model.fit_generator(trainDataGenerator, epochs=num_epochs, use_multiprocessing=True, workers=n_process)
    testData, testGender = get_data(test_split)
    testData = testData[:]
    testData = BasicSpatialDataset(testData, spatialMapping=spatialMapping)[:]
    testData = np.stack([datum[0] for datum in testData])
    testData=testData.reshape(*testData.shape, 1)
    y_pred = model.predict(testData)
    auc = roc_auc_score(testGender, y_pred.argmax())
    f1_score = f1_score(testGender, y_pred.argmax())
    accuracy = accuracy_score(testGender, y_pred.argmax())
    return {'test_scores': {
        'f1': f1_score,
        'acc': accuracy,
        'auc': auc
    }}
if __name__ == "__main__":
    ex.run_commandline()
