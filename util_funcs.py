import pickle as pkl
import json, os
import os.path as path
from addict import Dict
from pathos.multiprocessing import Pool
import pandas as pd
import numpy as np
import pymongo
import itertools
import pyedflib

def read_tse_file(path):
    tse_data_lines = []
    with open(path, 'r') as f:
        for line in f:
            if "#" in line:
                continue #this is a comment
            elif "version" in line:
                assert "tse_v1.0.0" in line #assert file is correct version
            elif len(line.strip()) == 0:
                continue #Just blank space, continue
            else:
                line = line.strip()
                subparts = line.split()
                tse_data_line = pd.Series(index=['start', 'end', 'label', 'p'])
                tse_data_line['start'] = float(subparts[0])
                tse_data_line['end'] = float(subparts[1])
                tse_data_line['label'] = str(subparts[2])
                tse_data_line['p'] = float(subparts[3])
                tse_data_lines.append(tse_data_line)
    tse_data = pd.concat(tse_data_lines, axis=1).T
    return tse_data

def read_tse_file_and_return_ts(path):
    pass #Not sure if I need this yet
    tse_data = read_tse_file(path)
    ann_types = get_annotation_types()


def edf_eeg_2_df(path):
    """ Transforms from EDF to pd.df, with channel labels as columns.
        This does not attempt to concatenate multiple time series but only takes
        a single edf filepath

    Parameters
    ----------
    path : str
        path of the edf file

    Returns
    -------
    pd.DataFrame
        index is time, columns is waveform channel label

    """
    reader = pyedflib.EdfReader(path)
    channel_names = [headerDict['label'] for headerDict in reader.getSignalHeaders()]
    sample_rates = [headerDict['sample_rate'] for headerDict in reader.getSignalHeaders()]
    start_time = reader.getStartdatetime()
    all_channels = []
    for i, channel_name in enumerate(channel_names):
        signal_data = reader.readSignal(i)
        signal_data = pd.Series(
            signal_data,
            index=pd.date_range(
                start=start_time,
                freq=pd.Timedelta(seconds=1/sample_rates[i]),
                periods=len(signal_data)
                ),
            name=channel_name
            )
        all_channels.append(signal_data)
    return pd.concat(all_channels, axis=1)


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

def get_patient_dir_names(data_split, ref, full_path=True):
    """ Gets the path names of the folders holding the patient info

    Parameters
    ----------
    data_split : str
        Which data split the patient belongs to
    ref : str
        which reference voltage system that is used
    full_path : bool
        Whether or not to return abs path

    Returns
    -------
    list
        list of strings describing path

    """
    assert data_split in get_data_split()
    assert ref in get_reference_node_types()
    p = Pool()
    root_dir_entry = data_split + "_" + ref
    config = read_config()
    root_dir_path = config[root_dir_entry]
    subdirs = get_abs_files(root_dir_path)
    patient_dirs = list(itertools.chain.from_iterable(p.map(get_abs_files, subdirs)))
    if full_path:
        return patient_dirs
    else:
        return [path.basename(patient_dir) for patient_dir in patient_dirs]

def get_session_dir_names(data_split, ref, full_path=True, patient_dirs=None):
    """Gets the path names of the folders holding the session info
        (patients can have multiple eeg sessions)

    Parameters
    ----------
    data_split : str
        Which data split the patient belongs to
    ref : str
        which reference voltage system that is used
    full_path : bool
        Whether or not to return abs path

    Returns
    -------
    list
        list of strings describing path
    """
    assert data_split in get_data_split()
    assert ref in get_reference_node_types()
    p = Pool()
    if patient_dirs is None:
        patient_dirs = get_patient_dir_names(data_split, ref)
    session_dirs = list(itertools.chain.from_iterable(p.map(get_abs_files, patient_dirs)))
    if full_path:
        return session_dirs
    else:
        return [path.basename(session_dir) for session_dir in session_dirs]

def get_all_token_file_names(data_split, ref, full_path=True):
    p = Pool()
    session_dirs = get_session_dir_names(data_split, ref)
    token_fns = list(itertools.chain.from_iterable(p.map(get_token_file_names, session_dirs)))
    if full_path:
        return token_fns
    else:
        return [path.basename(token_fn) for token_fn in token_fns]


def get_token_file_names(session_dir_path, full_path=True):
    sess_file_names = get_abs_files(session_dir_path)
    time_series_fns = [fn for fn in sess_file_names if fn[-4:] == '.edf']
    if full_path:
        return time_series_fns
    else:
        return [path.basename(fn) for fn in time_series_fns]

def get_session_data(session_dir_path):
    time_series_fns = get_token_file_names(session_dir_path)
    signal_dfs = []
    for fn in time_series_fns:
        signal_dfs.append(edf_eeg_2_df(fn))
    return pd.concat(signal_dfs)

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
