import sys, os
import tensorflow as tf
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
import data_reader as read
from multiprocessing import Process
import constants
import util_funcs
import functools
import pickle as pkl
import sacred
import ensembleReader as er
import constants
import random
import string
from time import time
from addict import Dict
import preprocessingV2.preprocessingV2 as ppv2
from functools import lru_cache


ex = sacred.Experiment(name="generate_tfrecords_from_cached_clean")
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))
@ex.config
def config():
    train_pkl_20s_index="/n/scratch2/ms994/medium_size/train/20sIndex.pkl"
    valid_pkl_20s_index="/n/scratch2/ms994/medium_size/valid/20sIndex.pkl"
    test_pkl_20s_index="/n/scratch2/ms994/medium_size/test/20sIndex.pkl"
    num_shards = 4
    overlap=2
    unit_size=4
    max_size=20
    super_seg_overlap=10
    run_all = True
    split_to_run = None
    file_pair_ind = None



@ex.capture
def get_data_file_reader(cachedIndex, split, directory, overlap, unit_size, max_size, super_seg_overlap):
    return ppv2.FileDataReader(split=split, directory=directory, cachedIndex=cachedIndex, overlap=overlap, unit_size=unit_size, max_size=max_size, super_seg_overlap=super_seg_overlap)

train_index = None
valid_index = None
test_index = None

@ex.capture
def get_train_index(train_pkl_20s_index):
    global train_index
    if train_index is not None:
        return train_index
    data = pkl.load(open(train_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    train_index = data
    return data


def split_index_into_pos_neg_class(index):
    newPositiveIndex = []
    newNegativeIndex = []
    for key in index.keys():
        index[key].original_ind = key
        if index[key].time_seizure_label.any():
            newPositiveIndex.append(index[key])
        else:
            newNegativeIndex.append(index[key])
    return newPositiveIndex, newNegativeIndex

@ex.capture
def get_valid_index(valid_pkl_20s_index):
    global valid_index
    if valid_index is not None:
        return valid_index
    data = pkl.load(open(valid_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    valid_index = data
    return data
@ex.capture
def get_test_index(test_pkl_20s_index):
    global test_index
    if test_index is not None:
        return test_index
    data = pkl.load(open(test_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    test_index = data
    return data

def train_generator(givenRange, trainDataset, trainIndexDict):
    def give_gen():
        for i in givenRange:
            print("{}/{}".format(i, len(givenRange)))
            yield get_data_from_index_datum(trainDataset, i, trainIndexDict[i]).SerializeToString()
    return give_gen
def valid_generator(givenRange, validDataset, validIndexDict):
    def give_gen():
        for i in givenRange:
            print("{}/{}".format(i, len(givenRange)))
            yield get_data_from_index_datum(validDataset, i, validIndexDict[i], is_train=False, split="valid").SerializeToString()
    return give_gen
def test_generator(givenRange, testDataset, testIndexDict):
    def give_gen():
        for i in givenRange:
            print("{}/{}".format(i, len(givenRange)))
            yield get_data_from_index_datum(testDataset, i, testIndexDict[i], is_train=False, split="test").SerializeToString()
    return give_gen

def create_train_class_dataset(index):
    return ppv2.FileDataReader(split="train", directory="/n/scratch2/ms994/medium_size/train", cachedIndex=index)

def getCachedData():
    testDR = ppv2.FileDataReader(split="test", directory="/n/scratch2/ms994/medium_size/test", cachedIndex=get_test_index())
    trainDR = ppv2.FileDataReader(split="train", directory="/n/scratch2/ms994/medium_size/train", cachedIndex=get_train_index())
    validDR = ppv2.FileDataReader(split="valid", directory="/n/scratch2/ms994/medium_size/valid", cachedIndex=get_valid_index())
    return trainDR, validDR, testDR


def get_data_from_index_datum(dataset, i, index_datum, is_train = True, split="train"):
    xData = dataset[i]
    yData = index_datum.time_seizure_label
    ySubtypeData = index_datum.time_seizure_subtypes
    split, patient, session, token = read.parse_edf_token_path_structure(index_datum.edf_file)
    montage_data = read.gen_seizure_channel_labels(index_datum.edf_file[:-4] + ".lbl", width=pd.Timedelta(seconds=2)).loc[pd.Timedelta(seconds=index_datum.start):pd.Timedelta(seconds=index_datum.start+20)]
    feature = { \
               'original_index': _int64_feature(index_datum.original_ind) if "original_ind" in index_datum.keys() else  _int64_feature(i) ,
               'data': _float_feature_list(xData[0].reshape(-1)), \
               'label': _int64_feature_list(yData.to_numpy().reshape(-1)), \
               'subtypeLabel': _int64_feature_list(ySubtypeData.to_numpy().reshape(-1)), \
               'patient': _int64_feature(read.getAllTrainPatients().index(patient) if is_train else 0), \
               'session': _int64_feature(read.getAllTrainSessions().index(session) if is_train else 0)
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# Helperfunctions to make your feature definition more readable
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

@ex.capture
def grab_datasets_files(dataset, indexDict, generator_func, split, num_shards):
    dataset_file_pairs = []
    min_shard_size = int(np.ceil(len(indexDict)/num_shards))
    for i in range(num_shards):
        dataset_file_pairs.append((tf.data.Dataset.from_generator(generator_func(range(min_shard_size*i, min(min_shard_size*(i+1), len(indexDict))), dataset,  indexDict), output_types = (tf.string), output_shapes=(tf.TensorShape([]))), "/n/scratch2/ms994/medium_size/{}_{}.tfrecords".format(split, i)))
    return dataset_file_pairs

def write(data, fn):
    writer = tf.data.experimental.TFRecordWriter(fn)
    writer.write(data)

@ex.capture
def writeAll(dataset_file_pairs):
    p = []
    for dataset, fileName in dataset_file_pairs:
        write(dataset,fileName)
    #     p.append(Process(target=write, args=(dataset, fileName)))
    # [process.start() for process in p]
    # [process.join() for process in p]

# Helperfunctions to make your feature definition more readable
def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

@ex.main
def main(run_all, split_to_run, file_pair_ind):
    trainDataset, validDataset, testDataset = getCachedData()
    trainIndexDict = get_train_index()
    train_dataset_file_pairs = grab_datasets_files(trainDataset, trainIndexDict, train_generator, "train")
    validIndexDict = get_valid_index()
    valid_dataset_file_pairs = grab_datasets_files(validDataset, validIndexDict, valid_generator, "valid")
    testIndexDict = get_test_index()
    test_dataset_file_pairs = grab_datasets_files(testDataset, testIndexDict, test_generator, "test")
    if run_all:
        writeAll(train_dataset_file_pairs)
        writeAll(valid_dataset_file_pairs)
        writeAll(test_dataset_file_pairs)
    if split_to_run == "train_positive_negative":
        positiveInd, negativeInd = split_index_into_pos_neg_class(trainIndexDict)
        positiveTrainData = create_train_class_dataset(positiveInd)
        negativeTrainData = create_train_class_dataset(negativeInd)
        positive_train_dataset_file_pairs = grab_datasets_files(positiveTrainData, positiveInd, train_generator, "train_pos")
        negative_train_dataset_file_pairs = grab_datasets_files(negativeTrainData, negativeInd, train_generator, "train_neg")
        write(positive_train_dataset_file_pairs[file_pair_ind][0], positive_train_dataset_file_pairs[file_pair_ind][1])
        write(negative_train_dataset_file_pairs[file_pair_ind][0], negative_train_dataset_file_pairs[file_pair_ind][1])

    if split_to_run == "train_negative":
        write(train_dataset_file_pairs[file_pair_ind][0], train_dataset_file_pairs[file_pair_ind][1])
    if split_to_run == "train":
        write(train_dataset_file_pairs[file_pair_ind][0], train_dataset_file_pairs[file_pair_ind][1])
    if split_to_run == "valid":
        write(valid_dataset_file_pairs[file_pair_ind][0], valid_dataset_file_pairs[file_pair_ind][1])
    if split_to_run == "test":
        write(test_dataset_file_pairs[file_pair_ind][0], test_dataset_file_pairs[file_pair_ind][1])



if __name__ == "__main__":
    ex.run_commandline()
