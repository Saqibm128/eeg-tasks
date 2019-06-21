import pickle as pkl
import json, os
import os.path as path
from addict import Dict
import pandas as pd
import numpy as np
import pymongo
import itertools
import pyedflib
from sacred.serializer import restore #to return a stored sacred result back

COMMON_DELTA = 1.0/256 #used for common resampling, inverse of sampling rate

def np_rolling_window(a, window):
    #https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_sacred_runs():
    return get_mongo_client().sacred.runs

def get_sacred_results(params):
    return restore(get_sacred_runs().find_one(params)['result'])

def get_abs_files(root_dir_path):
    """helper func to return full path names. helps with nested structure of
        extracted files

    Parameters
    ----------
    root_dir_path : type
        Description of parameter `root_dir_path`.

    Returns
    -------
    list
        Full paths of files, including directories, inside the root_dir_path
        If root_dir_path is a file and not a directory, this will fail

    """
    subdirs = os.listdir(root_dir_path)
    subdirs = [path.join(root_dir_path, subdir) for subdir in subdirs]
    return subdirs


cached_annotation_csv = None

def get_annotation_csv():
    global cached_annotation_csv
    if cached_annotation_csv is None:
        cached_annotation_csv = pd.read_csv(
            "data_labels.csv",
             header=0,
             dtype=str,
             keep_default_na=False,
             )
    return cached_annotation_csv

def get_annotation_types():
    """Used to get the specific annotation types. These are specified in
            .tse files and label specific time subsequences of the entire record
            These are also used in .lbl files to label time sequences of single channels

    Parameters
    ----------


    Returns
    -------
    list
        list of the lower case annotation codes

    """
    #https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.0/_DOCS/
    return get_annotation_csv()["class_code"].str.lower().tolist()

def get_data_split():
    return ["train", "dev_test"]

def get_reference_node_types():
    """The TUH dataset is further split based on what was the reference voltage
        See: https://www.isip.piconepress.com/publications/conference_proceedings/2016/ieee_spmb/montages/

    Parameters
    ----------


    Returns
    -------
    list
        strings representing the appropriate subdirectory that describes
        reference
    """
    return ["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"]


def get_mongo_client(path = "config.json"):
    '''
    Used for Sacred to record results
    '''
    config = read_config(path)
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)

def read_config(path="config.json"):
    return json.load(open(path, "rb"))


if __name__ == "__main__":
    print(read_config())
    print(get_annotation_types())
    print('spsw' in get_annotation_types())
