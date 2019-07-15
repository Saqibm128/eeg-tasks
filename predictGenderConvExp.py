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
import pickle as pkl
import sacred
ex = sacred.Experiment(name="gender_predict_conv")

@ex.named_config
def debug():
    max_length=1
    num_files=20
    batch_size=1

@ex.config
def config():
    train_split = "train"
    test_split = "dev_test"
    ref = "01_tcp_ar"
    n_process = 7
    num_files = None
    max_length = 2
    batch_size = 32



@ex.config
def get_simple_mapping_data(split, ref, n_process, num_files, max_length, batch_size):
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
    genderDict = cta.getGenderAndFileNames(split, ref)
    edfTokenPaths, genders = cta.demux_to_tokens(genderDict)
    if max_length is not None:
        max_length = pd.Timedelta(seconds=1) * max_length
    edfData = read.EdfDataset(split, ref, n_process=n_process, max_length=max_length)
    edfData.edf_tokens = edfTokenPaths
    genders = [1 if item[1]=='m' else 0 for item in genderDict]
    return EdfDataGenerator(edfData, num_classes=2, labels=np.array(genders), batch_size=batch_size)

@ex.main
def main(train_split):
    dataGenerator = get_simple_mapping_data(train_split)
    dataGenerator[0]

if __name__ == "__main__":
    ex.run_commandline()
