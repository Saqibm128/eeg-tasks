import pickle as pkl
import json, os
from addict import Dict
from pathos.multiprocessing import Pool
import pandas as pd
import numpy as np
import pymongo

def get_seizure_types():
    return ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ', 'MYSZ', 'SPSZ', 'TCSZ', 'TNSZ']
def get_mongo_client(path = "config.json"):
    config = read_config(path)
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)

def read_config(path="config.json"):
    return json.load(open(path, "rb"))

def read_preproc_1(id):
    root_path = read_config()["preprocessed_1"]
    return pkl.load(open(os.path.join(root_path, "seiz_{}.pkl".format(id)), "rb"));


if __name__ == "__main__":
    print(read_config())
    print(read_preproc_1(1))
    print(read_preproc_2(1))
    print(read_all_into_df(num_files=100))
