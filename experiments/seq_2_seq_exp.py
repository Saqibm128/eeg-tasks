import os
import sys
from keras_models import models
import pickle as pkl
import numpy as np
import pandas as pd
import data_reader as read
from copy import deepcopy as cp
import util_funcs
from sacred import Experiment
ex = Experiment(name="seq_2_seq_clustering")


@ex.config
def config():
    num_files = 2
    n_process = 8
    latent_dim = 100
    freq_bins = read.EdfFFTDatasetTransformer.freq_bins
    # num channels times number of freq bins we extrapolated out
    input_shape = 21 * len(freq_bins)
    window_size = 10  # seconds
    non_overlapping = True
    num_epochs = 10
    batch_size = 10
    precached_pkl = None
    precache = True
    validation_split = 0.2


@ex.capture
def get_data(
        n_process,
        num_files,
        window_size,
        non_overlapping,
        precache,
        precached_pkl):
    if precached_pkl is not None:
        return pkl.load(open(precached_pkl, "rb"))
    edfRawData = read.EdfDataset(
        "train",
        "01_tcp_ar",
        num_files=num_files,
        n_process=n_process)
    edfFFTData = read.EdfFFTDatasetTransformer(
        edfRawData,
        window_size=pd.Timedelta(
            seconds=window_size),
        non_overlapping=non_overlapping,
        precache=precache,
        n_process=n_process)
    seq2seqData = read.Seq2SeqFFTDataset(edfFFTData, n_process=n_process)
    return [edfRawData.edf_tokens, np.asarray(seq2seqData[:])]


@ex.capture
def create_model(input_shape, latent_dim):
    return models.get_seq_2_seq(input_shape, latent_dim)


@ex.main
def main():
    print("hi")
    data = get_data()
    model = create_model()
    model.fit([data, data], data)


if __name__ == "__main__":
    ex.run_commandline()
