import sys, os
sys.path.append(os.path.realpath(".."))
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"

import util_funcs
from importlib import reload
import data_reader as read
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import constants
import clinical_text_analysis as cta
import tsfresh
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from os import path
import predictSeizureConvExp as psce
import keras_models.dataGen as dg
from addict import Dict
reload(psce)
from keras.utils import multi_gpu_model
import keras.optimizers
import ensembleReader as er
import time
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, log_loss
from functools import lru_cache
from joblib import Parallel, delayed
import pickle as pkl
import gc
def get_pickle_no(start_seconds):
    return int(np.floor(start_seconds/1000)) + 1

def get_edf_pickle_name(edf, start_seconds, split):
    pickle_no = get_pickle_no(start_seconds)
    ref, patient, session, token = read.parse_edf_token_path_structure(edf)
    return f"/n/scratch2/ms994/medium_size/{split}/{patient}/{session}/{token}/intermediate_{pickle_no}"



# with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
@lru_cache(8)
def _get_pickle_data_lru_cache(basePath):
    return pkl.load(open(basePath, "rb"))
def get_pickle_data(edf, start_seconds, split="train"):
    #to keep all pickles manageable, we split for every 500 into temporary files
    basePath = get_edf_pickle_name(edf, start_seconds, split)
    return _get_pickle_data_lru_cache(basePath)
def get_data_from_start(edf, start_seconds, split="train"):
    data = get_pickle_data(edf, start_seconds, split)
    return data[start_seconds/2]["data"][:][0]

train_split_preprocessed = "/n/scratch2/ms994/medium_size/train"
test_split_preprocessed = "/n/scratch2/ms994/medium_size/test"
valid_split_preprocessed = "/n/scratch2/ms994/medium_size/valid"
train_split_preprocessed = "/n/scratch2/ms994/medium_size/train"
test_split_preprocessed = "/n/scratch2/ms994/medium_size/test"
valid_split_preprocessed = "/n/scratch2/ms994/medium_size/valid"

class Shuffler(util_funcs.MultiProcessingDataset):
    def __init__(self, data, num_shuffles_per_item=8):
        self.data = data
        self.n_process = 1
        self.num_shuffles_per_item = num_shuffles_per_item
    def __len__(self):
        return len(self.data) * self.num_shuffles_per_item
    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        oldI = int(np.floor(i / self.num_shuffles_per_item))
        datum = self.data[oldI]
        xDatum = datum[0]
        newXDatum = xDatum.copy()
        np.apply_along_axis(np.random.shuffle, 2, newXDatum)
        return newXDatum, datum[1]


class DataColator(util_funcs.MultiProcessingDataset):
    def __init__(self, data, file_size=500, collates=4):
        self.data = data
        self.collates = collates
        self.n_process=1
        self.file_size = file_size
        self.index_mapping = []
        numInd = 0
        round = 0
        while numInd < len(self.data):
            for i in range(collates):
                for j in range(file_size):
                    proposed_ind = i*file_size + j + round * file_size * collates
                    if proposed_ind < len(self.data):
                        self.index_mapping.append(proposed_ind)
                        numInd+=1
            round += 1


    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        new_i = self.index_mapping[i]
        return self.data[new_i]

class FileDataReader(util_funcs.MultiProcessingDataset):
    def __init__(self, cachedIndex=None, split="train", directory=train_split_preprocessed, train_label_files_segs=None, overlap=2, unit_size=4, max_size=20, super_seg_overlap=10, eagerLoad=False):
        self.directory = directory
        self.all_files = util_funcs.get_abs_files(util_funcs.get_abs_files(util_funcs.get_abs_files("/n/scratch2/ms994/medium_size/test/")), False)
        self.overlap = overlap
        self.super_seg_overlap = super_seg_overlap
        self.unit_size = unit_size
        self.eagerLoad = eagerLoad
        if train_label_files_segs is not None:
            self.train_label_files_segs = train_label_files_segs
        elif cachedIndex is None:

            def getFnAnn(tknFn):
                labelFile = read.read_tse_file(tknFn + "/label.tse")
                ann = er.generate_label_rolling_window(labelFile, pre_cooldown=0, post_cooldown=0,sample_time=0, num_seconds=2)
                return tknFn, ann
            fileNames = util_funcs.get_abs_files(util_funcs.get_abs_files(util_funcs.get_abs_files(directory)), False)
            self.train_label_files_segs = Parallel(4)([delayed(getFnAnn)(tknFn) for tknFn in fileNames])

        self.max_size = max_size
        self.split = split
        self.use_mp = False
        if cachedIndex is None:
            self.indexDict = Dict() #used to grab and set the indexes used to grab data from the fs
            currentInd = 0
            for i in range(len(train_label_files_segs)):
                max_segment_index = train_label_files_segs[i][1].index.max()/pd.Timedelta(seconds=self.overlap)
                for j in range(int(np.floor((max_segment_index - max_size)/super_seg_overlap))):
                    startTime = j * self.max_size
                    self.indexDict[currentInd].start = startTime
                    self.indexDict[currentInd].edf_file = train_label_files_segs[i][0]

                    labelSlice = train_label_files_segs[i][1][ \
                                                              pd.Timedelta(seconds=startTime): \
                                                              pd.Timedelta(seconds=startTime)+pd.Timedelta(seconds=self.max_size) \
                                                             ].iloc[:-1]
                    self.indexDict[currentInd].label = not (labelSlice == "bckg").all()
                    self.indexDict[currentInd].time_seizure_label =  (labelSlice != "bckg")
                    self.indexDict[currentInd].time_seizure_subtypes = labelSlice.apply(lambda x: constants.SEIZURE_SUBTYPES.index(x))
                    currentInd+=1
            # pkl.dump(self.indexDict, open("/n/scratch2/ms994/medium_size/"+split + "/20sindex.pkl", "wb"))
        else:
            self.indexDict = cachedIndex
    def __len__(self):
        return len(self.indexDict)
    def  __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        label = self.indexDict[i].time_seizure_label
        length = int(self.max_size / self.overlap - self.unit_size / self.overlap + 1)
        data = np.ndarray((length,21,1000))
        for j in range(length):
            data[j] = get_data_from_start(self.indexDict[i].edf_file, self.indexDict[i].start + j * 2, split=self.split)
        return data, label.values

# class SeizureOnlyDataReader(util_funcs.MultiProcessingDataset):
#     def __init__(self,):

class RULDataReader(util_funcs.MultiProcessingDataset):
    def __init__(self, cachedIndex=None, split="train", force_file_sort=False):
        self.indexDict = cachedIndex
        self.force_file_sort = force_file_sort #since we use an lru_cache, lets just grab segments from same file while we still can
        self.rebalance()
        self.split = split
        self.use_mp = False
    def rebalance(self):
        oldIndicesByLabels = Dict()
        allLabels = Dict()
        for i in range(len(self.indexDict)):
            label = self.indexDict[i].label #use the first label
            if label not in oldIndicesByLabels.keys():
                oldIndicesByLabels[label] = []
                allLabels[label] = 0
            oldIndicesByLabels[label].append(i)
            allLabels[label] += 1

        min_label_count = min([allLabels[label] for label in allLabels.keys()])
        self.list_IDs = []
        for label in oldIndicesByLabels.keys():
            oldIndicesByLabels[label] = np.random.choice(oldIndicesByLabels[label], size=min_label_count, replace=False)
            for oldInd in oldIndicesByLabels[label]:
                self.list_IDs.append(oldInd)
        if self.force_file_sort:
            temp_list_IDs = self.list_IDs
            self.list_IDs = []
            file_to_list_id_mapping = Dict()

            for listId in temp_list_IDs:
                if not self.indexDict[listId].edf_file in file_to_list_id_mapping.keys():
                    file_to_list_id_mapping[self.indexDict[listId].edf_file] = []
                file_to_list_id_mapping[self.indexDict[listId].edf_file].append(listId)
            for listId in file_to_list_id_mapping.keys():
                self.list_IDs += file_to_list_id_mapping[listId]
        else:
            np.random.shuffle(self.list_IDs)
    def __len__(self):
        return len(self.list_IDs)
    def  __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        new_i = self.list_IDs[i]
        label = self.indexDict[new_i].time_seizure_label
        data = np.ndarray((11,21,1000))
        for j in range(11):
            data[j] = get_data_from_start(self.indexDict[new_i].edf_file, self.indexDict[new_i].start + j * 2, split=self.split)
        return data, label.values
