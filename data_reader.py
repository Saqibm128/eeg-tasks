import pandas as pd
import numpy as np
import itertools
import pyedflib
from os import path
from util_funcs import read_config, get_abs_files, get_annotation_types, get_data_split, get_reference_node_types, COMMON_FREQ
from multiprocessing import Manager, Process


# dataset = Dataset(num_files=10)
# eeg_df_256_hz, labels_by_256_hz = Dataset().get(i)
# model


#TODO: no way we can load all the data at once
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class EdfFFTDatasetTransformer():
    """Implements an indexable dataset applying fft to entire timeseries,
        returning histogram bins of fft frequencies

    Parameters
    ----------
    edf_dataset : EdfDataset
        an array-like returning the raw channel data and the output as a tuple

    Attributes
    ----------
    edf_dataset

    """
    def __init__(self, edf_dataset):
        self.edf_dataset = edf_dataset
    def __len__(self):
        return len(self.edf_dataset)
    def __getitem__(self, i):
        original_data = self.edf_dataset[i]
        fft_data = np.abs(np.fft.fft(original_data[0].values))
        fft_freq = np.fft.fftfreq(fft_data.shape[0], d=COMMON_FREQ)
        fft_freq_bins = list(range(50))
        new_fft_hist = pd.DataFrame(index=fft_freq_bins[:-1], columns=original_data[0].columns)
        for i, name in enumerate(original_data[0].columns):
            new_fft_hist[name] = np.histogram(fft_freq, bins=fft_freq_bins, weights=fft_data[:,i])[0]
        return new_fft_hist, original_data[1]

class EdfDataset():
    """Short summary.

    Parameters
    ----------
    data_split : type
        Description of parameter `data_split`.
    ref : type
        Description of parameter `ref`.
    resample : type
        Description of parameter `resample`.

    Attributes
    ----------
    manager : multiprocessing.Manager
        used to manage multiprocessing. TODO: not implemented
    edf_tokens : list
        a list of edf file paths to consider, assumes a corresponding tse file
        exists
    data_split : str
    ref : str
    resample : pd.Timedelta

    """
    def __init__(self, data_split, ref, resample=pd.Timedelta(seconds=COMMON_FREQ)):
        self.data_split = data_split
        self.ref = ref
        self.resample = resample
        self.manager = Manager()
        self.edf_tokens = get_all_token_file_names(data_split, ref)
    def __len__(self):
        return len(self.edf_tokens)
    def __getitem__(self, i):
        return get_edf_data_and_label_ts_format(self.edf_tokens[i], self.resample)
    #
    # def get_data_runner(to_get_queue, to_return_queue):
    #     for edf_path in iter(to_get_queue.get, None):
    #         to_return_queue = ()
    # def get_data_multiprocess():


def get_edf_data_and_label_ts_format(edf_path, resample=pd.Timedelta(seconds=COMMON_FREQ)):
    edf_data = edf_eeg_2_df(edf_path, resample)
    tse_data_path = convert_edf_path_to_tse(edf_path)
    tse_data_ts = read_tse_file_and_return_ts(tse_data_path, edf_data.index)
    return edf_data, tse_data_ts

def read_tse_file(tse_path):
    tse_data_lines = []
    with open(tse_path, 'r') as f:
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

def convert_edf_path_to_tse(edf_path):
    return edf_path[:-4] + ".tse"

def read_tse_file_and_return_ts(tse_path, ts_index):
    ann_y = read_tse_file(tse_path)
    ann_y_t = pd.DataFrame(columns=get_annotation_types(), index=ts_index)
    ann_y.apply(lambda row: ann_y_t[row['label']].loc[pd.Timedelta(seconds=row['start']):pd.Timedelta(seconds=row['end'])].fillna(row['p'], inplace=True), axis=1)
    ann_y_t.fillna(0, inplace=True)
    return ann_y_t


def edf_eeg_2_df(path, resample=None):
    """ Transforms from EDF to pd.df, with channel labels as columns.
        This does not attempt to concatenate multiple time series but only takes
        a single edf filepath

    Parameters
    ----------
    path : str
        path of the edf file

    resample : pd.Timedelta
        if None, returns original data with original sampling
        otherwise, resamples to correct Timedelta using forward filling

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
    data = pd.concat(all_channels, axis=1)
    data.index = data.index - data.index[0]
    if resample != None:
        data = data.resample(resample).ffill()
    return data

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
    root_dir_entry = data_split + "_" + ref
    config = read_config()
    root_dir_path = config[root_dir_entry]
    subdirs = get_abs_files(root_dir_path)
    patient_dirs = list(itertools.chain.from_iterable([get_abs_files(subdir) for subdir in subdirs]))
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
    if patient_dirs is None:
        patient_dirs = get_patient_dir_names(data_split, ref)
    session_dirs = list(itertools.chain.from_iterable([get_abs_files(patient_dir) for patient_dir in patient_dirs]))
    if full_path:
        return session_dirs
    else:
        return [path.basename(session_dir) for session_dir in session_dirs]

def get_all_token_file_names(data_split, ref, full_path=True):
    session_dirs = get_session_dir_names(data_split, ref)
    token_fns = list(itertools.chain.from_iterable([get_token_file_names(session_dir) for session_dir in session_dirs]))
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
