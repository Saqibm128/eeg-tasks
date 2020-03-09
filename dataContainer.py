# represents a V2 of sorts, while still using similar data processing
import pandas as pd
import numpy as np
import itertools
import pyedflib
from os import path
import sys, os
import util_funcs
from util_funcs import read_config, get_abs_files, get_annotation_types, get_data_split, get_reference_node_types, np_rolling_window
import multiprocessing as mp
import argparse
import pickle as pkl
import constants
import re
from scipy.signal import butter, lfilter
import pywt
from wf_analysis import filters
from addict import Dict
import functools
from copy import deepcopy

class DataContainerV2(util_funcs.MultiProcessingDataset):
    def __init__(self):
        self.transformers = [] #itself a list of data containers, each with similar functions
        pass
    def __len__(self):
        return 0
    def __getitem__(self, i):
        return None
    def get_main_label(self, i):
        pass
    def transform(self):
        return None
    def reportStats(self):
        '''
        Returns a set of statistics about each step in the pipeline
        '''
        return None
    def attachStatsReporter(self, reporter):
        self.reporters.append(reporter)
        return None
    def attachNewDataTransformer(self, transformer):
        self.transformers.append(transformer)
