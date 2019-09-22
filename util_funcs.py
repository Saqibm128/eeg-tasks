import pickle as pkl
import json
import os
import os.path as path
import pandas as pd
import numpy as np
import pymongo
import itertools
import pyedflib
from sacred.serializer import restore  # to return a stored sacred result back
import multiprocessing as mp
import queue
import constants
from functools import lru_cache
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


root_path = "/home/ms994/" if "EEG_ROOT" not in os.environ.keys() else os.environ["EEG_ROOT"]

class ImbalancedClassResampler():
    SMOTE = "SMOTE"
    RANDOM_UNDERSAMPLE = "RANDOM_UNDERSAMPLE"
    def __init__(self, method=None, n_process=1):
        self.method = method
        self.n_process = n_process
        self.resampler = None

    def fit(self, x, y):
        if self.method is None:
            return self
        if self.method == ImbalancedClassResampler.SMOTE:
            self.resampler = SMOTE(n_jobs=self.n_process)
            self.resampler.fit(x, y)
        elif self.method == ImbalancedClassResampler.RANDOM_UNDERSAMPLE:
            self.resampler = RandomUnderSampler()

    def get_params(self,deep):
        return {"method": self.method}

    def set_params(self, method):
        self.method = method

    def resample(self,x, y):
        if self.method is None:
            return x, y
        return self.resampler.resample(x, y)

    def fit_resample(self,x, y):
        self.fit(x, y)
        return self.transform(x, y)

class MultiProcessingDataset():
    """Class to help improve speed of looking up multiple records at once using multiple processes.
        Was originally going to be designed around batch loading in, but was just used as a way to more quickly
        populate an array-like into memory

            Just make this the parent class, then call the getItemSlice method on slice objects
        Issues:
            Doesn't solve original problem of being optimized for keras batches, only solves
                the fact that I needed some dataset that could quickly use multiple cores to
                get data. Use the models in keras_models.dataGen
            SLURM opaquely kills processes if it consume too much memory, so we gotta
                double check and see that there are placeholders in the toReturn array left
            The toReturn array uses integer placeholders (representing logical indices of the
                dataset ). If the returning datatype returned by indexing is also
                an integer, then this won't work
            Recovery from OOM is single threaded. Maybe we wanna make this
                use mp if this becomes a new bottleneck?


    """
    def should_use_mp(self, i):
        return type(i) == slice

    def should_use_mp(self, i):
        return type(i) == slice or type(i) == list

    def getItemSlice(self, i):
        #assign index as placeholder for result in toReturn
        if type(i) == slice:
            placeholder = [j for j in range(*i.indices(len(self)))] #use to look up correct index because using the ".index" method in an array holding arrays leads to comparison error
            toReturn = [j for j in range(*i.indices(len(self)))]
        elif type(i) == list: #indexing by list
            placeholder = [j for j in i]
            toReturn = [j for j in i]
        if hasattr(self, "use_mp") and self.use_mp == False: #in case it makes more sense to just use a loop instead of dealing with overhead of starting processes
            for i, j in enumerate(toReturn):
                toReturn[i] = self[j]
            return toReturn
        manager = mp.Manager()
        inQ = manager.Queue()
        outQ = manager.Queue()
        if self.n_process > 1: #otherwise use for loop
            [inQ.put(j) for j in toReturn]
            [inQ.put(None) for j in range(self.n_process)]
            processes = [
                mp.Process(
                    target=self.helper_process,
                    args=(
                        inQ,
                        outQ)) for j in range(
                    self.n_process)]
            if not hasattr(self, "verbose") or self.verbose == True:
                print("Starting {} processes".format(self.n_process))
            [p.start() for p in processes]
            [p.join() for p in processes]
            startIndex = toReturn[0]
        while not outQ.empty():
            place, res = outQ.get()
            index = placeholder.index(place)
            if type(res) == int:
                if not hasattr(self, "verbose") or self.verbose == True:
                    print("SLURM sent OOM event, retrying: ", res)
                res = self[place] #slurm sent oom event, we gotta try again.
            toReturn[index] = res
        for index, res in enumerate(toReturn):
            if type(res) == int:
                toReturn[index] = self[res]
        return toReturn
        # return Pool().map(self.__getitem__, toReturn)

    def helper_process(self, in_q, out_q):
        for i in iter(in_q.get, None):
            if not hasattr(self, "verbose") or self.verbose == True:
                if not hasattr(self, "verbosity"):
                    self.verbosity = 10
                if i % self.verbosity == 0:
                    print("retrieving: {}".format(i))
            out_q.put((i, self[i]))




def np_rolling_window(a, window):
    # https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
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



@lru_cache(10)
def get_common_channel_names(): #21 channels in all edf datafiles
    cached_channel_names = list(
        pd.read_csv(
            path.join(root_path,"dbmi_eeg_clustering/assets/channel_names.csv"),
            header=None)[1])
    return cached_channel_names

@lru_cache(10)
def get_file_sizes(split, ref):
    assert split in get_data_split()
    assert ref in get_reference_node_types()
    return pd.read_csv(path.join(root_path, "dbmi_eeg_clustering/assets/{}_{}_file_lengths.csv".format(split, ref)), header=None, index_col=[0])


@lru_cache(10)
def get_annotation_csv():
    cached_annotation_csv = pd.read_csv(
        path.join(
        root_path,"dbmi_eeg_clustering/assets/data_labels.csv"),
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
    # https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.0/_DOCS/
    return get_annotation_csv()["class_code"].str.lower().tolist()


def get_data_split():
    return ["train", "dev_test", "combined", None]


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


def get_mongo_client(path=path.join(root_path,"dbmi_eeg_clustering/config.json")):
    '''
    Used for Sacred to record results
    '''
    config = read_config(path)
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)

config = None #don't use lru_cache, instead this is exposed for idiots like me to mess with
def switch_to_seizure_subset():
    global config
    config = read_config()
    config.update(config["seizure_config"])
    return config

def switch_to_general_superset():
    global config
    config = read_config()
    config.update(config["tuh_eeg_all"])
    return config

def read_config(path=path.join(root_path,"dbmi_eeg_clustering/config.json")):
    global config
    if config is None:
        config = json.load(open(path, "rb"))
    return config


if __name__ == "__main__":
    print(read_config())
    print(get_annotation_types())
    print('spsw' in get_annotation_types())
