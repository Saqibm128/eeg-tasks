import util_funcs
import clinical_text_analysis as cta
import data_reader as read
from wf_analysis import filters
import pandas as pd
import numpy as np
import constants
import multiprocessing as mp
from addict import Dict

class EnsemblerSequence():
    '''
    Like keras_models.dataGen.EdfDataGenerator but takes into account the ordering
    of the token files
    '''
    def __init__(
        self,
        ensembler,
        batch_size
    ):
        assert type(ensembler) == EdfDatasetEnsembler
        self.ensembler = ensembler
        self.ensembledData = ensembler[:]
        self.batch_size = batch_size
        self.sampleInfo = ensembler.sampleInfo
        self.seqIndex = Dict()
        createSeqIndex()

    def createSeqIndex(self):
        pass

    def __len__(self):
        return np.ceil(len(self.ensembler)/self.batch_size)

    def __getitem__(self, i):
        pass

    def on_epoch_end(self):
        pass

class EdfDatasetEnsembler(util_funcs.MultiProcessingDataset):
    """
    Similar to EdfDataset but allows for multiple sampling from the same dataset (i.e. make multiple instances from the same edf token file)
    """
    #currently only mode that is accepted, maybe we can try some kinda complete ensemble (i.e. use all possible non_overlappings)
    RANDOM_SAMPLE_ENSEMBLE = 'RANDOM_SAMPLE_ENSEMBLE'
    def __init__(
            self,
            data_split,
            ref,
            num_files=None,
            resample=pd.Timedelta(
                seconds=constants.COMMON_DELTA),
            max_length=pd.Timedelta(seconds=4),
            expand_tse=False, #Save memory, don't try to make time by annotation df
            dtype=np.float32,
            n_process=None,
            use_average_ref_names=True,
            filter=True,
            lp_cutoff=30,
            hp_cutoff=(constants.COMMON_FREQ/2-2), #get close to nyq without actually hitting it to avoid errors
            order_filt=5,
            columns_to_use=util_funcs.get_common_channel_names(),
            use_numpy=True,
            ensemble_mode=RANDOM_SAMPLE_ENSEMBLE,
            max_num_samples=20,
            file_lengths=None, #automatically populated if not given
            edf_tokens=None,
            labels=None, # labels that map to edf token level
            generate_sample_info=True
            ):
        if labels is not None:
            assert len(labels) == len(edf_tokens)
        self.data_split = data_split
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.ref = ref
        self.resample = resample
        self.dtype = dtype
        if (type(max_length) == int):
            max_length = max_length * pd.Timedelta(seconds=pd.Timedelta(constants.COMMON_DELTA))
        self.max_length = max_length
        self.manager = mp.Manager()
        if edf_tokens is None:
            self.edf_tokens = read.get_all_token_file_names(data_split, ref)
        else:
            self.edf_tokens = edf_tokens
        self.expand_tse = expand_tse
        self.use_average_ref_names = use_average_ref_names
        if num_files is not None:
            self.edf_tokens = self.edf_tokens[0:num_files]
        self.filter = filter
        self.hp_cutoff = hp_cutoff
        self.lp_cutoff = lp_cutoff
        self.order_filt = order_filt
        self.columns_to_use = columns_to_use
        self.use_numpy = use_numpy
        self.ensemble_mode = ensemble_mode
        self.max_num_samples = max_num_samples
        if file_lengths is None:
            file_lengths = util_funcs.get_file_sizes(data_split, ref)
        self.file_lengths=file_lengths
        self.labels = labels




        self.sampleInfo=Dict()
        if generate_sample_info:
            self.generateSampleInfo()

    def generateSampleInfo(self):
            currentIndex = 0
            if self.ensemble_mode == EdfDatasetEnsembler.RANDOM_SAMPLE_ENSEMBLE:
                for i, token_file in enumerate(self.edf_tokens):
                    totalNumExtractable = int(np.floor(self.file_lengths.loc[token_file] * pd.Timedelta(seconds=1) /self.max_length))
                    max_num_samples = min(self.max_num_samples, totalNumExtractable) #if file is smaller than max_num_samples * max_length, then we can't extract as many samples
                    chosen_samples = np.random.choice(totalNumExtractable, size=max_num_samples, replace=False)
                    for j, sample_in_token in enumerate(chosen_samples):
                        self.sampleInfo[currentIndex].token_file_path = token_file
                        self.sampleInfo[currentIndex].sample_num = sample_in_token
                        self.sampleInfo[currentIndex].sample_width = self.max_length
                        self.sampleInfo[currentIndex].token_file_index = i
                        if self.labels is not None:
                            self.sampleInfo[currentIndex].label = self.labels[i]


                        currentIndex+=1
            else:
                raise Exception("ensemble_mode {} not implemented".format(self.ensemble_mode))




    ENSEMBLE_PREDICTION_OVER_EACH_SAMP = "average_over_each_samp"
    ENSEMBLE_PREDICTION_EQUAL_VOTE = "equal_vote"
    ENSEMBLE_PREDICTION_MODES = [ENSEMBLE_PREDICTION_EQUAL_VOTE, ENSEMBLE_PREDICTION_OVER_EACH_SAMP]
    def getEnsemblePrediction(self, pred_labels, mode=ENSEMBLE_PREDICTION_OVER_EACH_SAMP):
        """
        Given an n by len(self.sampleInfo) array of predicted labels, get an average
        of all predictions for a given edf token file, such that it can be compared
        to self.label.


        Parameters
        ----------
        pred_labels : ndarray
            array of dim n_classes by n_instances
        mode : str
            describes how prediction should be done

        Returns
        -------
        ndarray
            of 2 by len(self.edf_tokens), first index is True labels, second is average prediction

        """
        assert mode in EdfDatasetEnsembler.ENSEMBLE_PREDICTION_MODES
        if mode == EdfDatasetEnsembler.ENSEMBLE_PREDICTION_EQUAL_VOTE:
            pred_labels = pred_labels.argmax(1)
        pred_vs_true = Dict()
        for i in range(len(self)):
            tokenFile = self.sampleInfo[i].token_file_path
            if tokenFile not in pred_vs_true.keys():
                pred_vs_true[tokenFile].trueLabel = []
                pred_vs_true[tokenFile].predLabel = []
            pred_vs_true[tokenFile].trueLabel.append(self.sampleInfo[i].label)
            pred_vs_true[tokenFile].predLabel.append(pred_labels[i])
        toReturn = []
        for tokenFile in pred_vs_true.keys():
            if mode == EdfDatasetEnsembler.ENSEMBLE_PREDICTION_OVER_EACH_SAMP:
                toReturn.append((np.mean(pred_vs_true[tokenFile].trueLabel), np.mean(pred_vs_true[tokenFile].predLabel, axis=0).argmax()))
            elif mode == EdfDatasetEnsembler.ENSEMBLE_PREDICTION_EQUAL_VOTE:
                toReturn.append((np.mean(pred_vs_true[tokenFile].trueLabel), (np.mean(pred_vs_true[tokenFile].predLabel))))
        return np.array(toReturn).transpose() #becomes 2 by n array, where first array is the true label, the second is the predLabel

    def getEnsembledLabels(self):
        labels = []
        for i in range(len(self)):
            labels.append(self.sampleInfo[i].label)
        return np.array(labels)


    def __len__(self):
        return len(self.sampleInfo)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        indexData = self.sampleInfo[i]
        data = read.edf_eeg_2_df(indexData.token_file_path, resample=self.resample, start=pd.Timedelta(indexData.sample_num * self.max_length), max_length=self.max_length)
        if (self.max_length != None and max(data.index) > self.max_length):
            if type(self.max_length) == pd.Timedelta:
                data = data.loc[pd.Timedelta(seconds=0):self.max_length].iloc[0:-1]
            else:
                data = data.iloc[0:self.max_length]
        if self.use_average_ref_names:
            data = data[self.columns_to_use]
        if self.filter:
            data = data.apply(
                lambda col: filters.butter_bandgap_filter(
                    col,
                    lowcut=self.lp_cutoff,
                    highcut=self.hp_cutoff,
                    fs=pd.Timedelta(
                        seconds=1) /
                    self.resample,
                    order=self.order_filt),
                axis=0)
        data = data.fillna(method="ffill").fillna(method="bfill")
        if self.use_numpy:
            data = data.values
        if "label" not in indexData.keys():
            return data
        else:
            return data, indexData.label

class AdditionalLabelEndpoints():
    def __init__(self, split, ref, sampleInfo=None, ensembler=None):
        assert sampleInfo is not None or ensembler is not None
        self.split = split
        self.ref = ref
        self.sampleInfo = sampleInfo
        if self.sampleInfo is None:
            self.sampleInfo = ensembler.sampleInfo
    def get_genders(self):
        genderDictItems = cta.demux_to_tokens(cta.getGenderAndFileNames(self.split, self.ref, convert_gender_to_num=True))
        genderDict = {}
        for index in range(len(genderDictItems)):
            key = genderDictItems[0][index]
            val = genderDictItems[1][index]

            genderDict[key] = val
        genders = []
        for i in range(len(self.sampleInfo)):
            tokenFile = self.sampleInfo[i].token_file_path
            genders.append(genderDict[tokenFile])
            self.sampleInfo[i].label = genderDict[tokenFile]
        return genders
    def get_ages(self):
        agesDictItems = cta.demux_to_tokens(cta.getAgesAndFileNames(self.split, self.ref))
        agesDict = {}
        for index in range(len(agesDictItems)):
            key = agesDictItems[0][index]
            val = agesDictItems[1][index]
            agesDict[key] = val
        ages = []
        for i in range(len(self.sampleInfo)):
            tokenFile = self.sampleInfo[i].token_file_path
            ages.append(agesDict[tokenFile])
            self.sampleInfo[i].label = agesDict[tokenFile]
        return ages
