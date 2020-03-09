import sacred
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
import data_reader as read
from keras import backend as K
# from multiprocessing import Process
import constants
import util_funcs
import functools
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels, RULEdfDataGenerator, RULDataGenMultipleLabels
import pickle as pkl
import ensembleReader as er
import random
import string
import mne
import pyprep
from pyprep import PrepPipeline
from pathlib import Path
import shutil
import time
import pickle
from joblib import Parallel, delayed


ex = sacred.Experiment(name="generate_cached_clean")
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

def run_prep(file_name, annotation, split="train"):
        data = mne.io.read_raw_edf(file_name, preload=True)
        data = data.pick_channels(util_funcs.get_common_channel_names())
        data = data.reorder_channels(util_funcs.get_common_channel_names())
        data.rename_channels(constants.MNE_CHANNEL_EDF_MAPPING)
        data.resample(512) #upsample to highest frequency, as per best practice
        data.set_eeg_reference()

        data.set_montage("standard_1020")
        data.filter(1, 50)

        montage_kind = "standard_1020"
        maxTime = annotation.index.max()/pd.Timedelta(seconds=1)
        montage = mne.channels.make_standard_montage(montage_kind)
        ref, patient, session, token = read.parse_edf_token_path_structure(file_name)

        # for i in range(int(maxTime/2)):
        basePath = f"/n/scratch2/ms994/medium_size/{split}/{patient}/{session}/{token}/"
        Path(basePath).mkdir( parents=True, exist_ok=True)


        shutil.copyfile(file_name[:-4]+".tse", f"{basePath}label.tse")
        shutil.copyfile(file_name[:-4]+".lbl", f"{basePath}montage.lbl")
        shutil.copyfile(file_name[:-9]+".txt", f"{basePath}notes.txt")

        dataDict = Dict()

        for i in range(int(maxTime/2) - 1):
            croppedData = data.copy().crop(i*2, i*2 + 4)
            croppedData.resample(constants.COMMON_FREQ) #resample to minimum
            dataDict[i].index = i
            dataDict[i].data = croppedData
            dataDict[i].start = i*2
            dataDict[i].end = i*2 + 4
            if (i % 500 == 499): # save up to 500 separate data segments at a time to avoid IO bottleneck in scratch2, but also to avoid creating any pickle that is too big to parse_edf_token_path_structure
                pickle.dump(dataDict, open(basePath+f"intermediate_{int(np.ceil(i/500))}", "wb"))
                dataDict = Dict()
        pickle.dump(dataDict, open(basePath+f"intermediate_{int(np.ceil(i/500))}", "wb"))
        print(f"COMPLETED {file_name}")
        # print("hi")
        # raise Exception()
            # prep_params = {'ref_chs': data.ch_names,
            #                'reref_chs': data.ch_names,
            #                'line_freqs': np.arange(60, croppedData.info["sfreq"]/2, 60)}
            # prep = pyprep.PrepPipeline(croppedData, prep_params, montage, ransac=False)
            # try:
            #     prep.fit()
            #     prep.raw.resample(constants.COMMON_FREQ) #downsample to common freq
            #     prep.raw.save(basePath+f"start_{i*2}_end_{i*2+4}.raw.fif", overwrite=True)
            # except:
            #     print("failed to run prep, data segment was too noisy")

@ex.main
def main():
    start = time.time()
    mne.cuda.init_cuda() #try to initialize cuda device
    train_split_path = "/home/ms994/v1.5.0/edf/train/"
    test_split_path = "/home/ms994/v1.5.0/edf/dev_test/"

    eds = er.EdfDatasetSegments(pre_cooldown=0, post_cooldown=0, sample_time=0, num_seconds=1, n_process=20)



    train_label_files_segs = eds.get_train_split()

    #debug line
    run_prep(train_label_files_segs[0][0], train_label_files_segs[0][1], split="train")


    n_jobs = 6
    Parallel(n_jobs)([delayed(run_prep)(train_label_files_segs[i][0], train_label_files_segs[i][1], split="train") for i in range(len(train_label_files_segs))])
    valid_label_files_segs = eds.get_valid_split()
    Parallel(n_jobs)([delayed(run_prep)(valid_label_files_segs[i][0], valid_label_files_segs[i][1], split="valid") for i in range(len(valid_label_files_segs))])
    test_label_files_segs = eds.get_test_split()
    Parallel(n_jobs)([delayed(run_prep)(test_label_files_segs[i][0], test_label_files_segs[i][1], split="test") for i in range(len(test_label_files_segs))])


    print(f"took {(time.time() - start)/60} minutes")


if __name__ == "__main__":
    ex.run_commandline()
    # main()
