import sacred
ex = sacred.Experiment(name="age_learning_exp")

#Sanity check to see if we can do something fundamental like this

import data_reader as read
import util_funcs
import pandas as pd
import numpy as np
from os import path

@ex.config
def config():
    ref = "01_tcp_ar"
    split = "train"
    n_process = 6
    precache = True
    num_files = None

@ex.capture
def get_data(split, ref, n_process, precache, num_files):
    ageData = read.getAgesAndFileNames(split, ref)
    if num_files is not None:
        ageData = ageData[0:num_files]
    session_files = [ageDatum[0] for ageDatum in ageData]
    ages = [ageDatum[1] for ageDatum in ageData]

    #associate first token file with each session for now
    tokenFiles = []
    for session_file in session_files:
        session_dir = path.dirname(session_file)
        session_tkn_files = read.get_token_file_names(session_dir)
        session_tkn_files.sort()
        tokenFiles.append(session_tkn_files[0])
    edfReader = read.EdfDataset(split, ref, expand_tse=False) #discarding the annotations eventually
    edfReader.edf_tokens = tokenFiles #override to use only eegs with ages we have
    edfFFTData = read.EdfFFTDatasetTransformer(edf_dataset=edfReader, n_process=n_process, precache=True)
    return edfFFTData, ages


@ex.main
def main():
    data = get_data()
    raise Exception()
    print("hi")

if __name__ == "__main__":
    ex.run_commandline()
