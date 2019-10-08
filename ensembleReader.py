import util_funcs
import clinical_text_analysis as cta
import data_reader as read
from wf_analysis import filters
import pandas as pd
import numpy as np
import constants
import multiprocessing as mp
from addict import Dict
from sklearn.model_selection import train_test_split
import random

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

class Utility_Custom_Annotater(util_funcs.MultiProcessingDataset):
    def __init__(self, files, labels, custom_annotate, n_process=20):
        self.labels = labels
        self.files = files
        self.n_process = n_process
        self.verbosity = 1000
        self.custom_annotate = custom_annotate
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            print("starting custom_annotater")
            return self.getItemSlice(i)
        return self.files[i], self.custom_annotate(self.labels[i])

class EdfDatasetSegments():
    """Short summary.

    bckg | sample | preseizure | seizure | postsz | bckg
    arrangement of the allocations of the segments for each seizure

    Parameters
    ----------
    train_split : string
        based on tuh, train or dev_test
    test_split : string
        based on tuh, train or dev_test
    ref : string
        reference system from tuh, usually 01_tcp_ar
    n_process : int
        num processes to launch
    valid_size : double
        how much of train to reserve for validation set
    pre_cooldown : int
        number of seconds to wait before a seizure occurs before sampling section
    post_cooldown : int
        number of secs after seizure before we can allow another sample to start
    sample_time : int
        max size of sampling before a seizure
    num_seconds : int
        granularity of label
    use_rolling : bool
        use the generate_label_rolling_window function (which is slow)

    Attributes
    ----------

    """
    def __init__(self,
                 train_split="train",
                 test_split="dev_test",
                 ref="01_tcp_ar",
                 n_process=20,
                 valid_size=0.2,
                 pre_cooldown=5,
                 post_cooldown=None,
                 sample_time=60,
                 num_seconds=4,
                 use_rolling=False):
        self.train_split = train_split
        self.test_split = test_split
        self.ref = ref
        self.train_valid_slr = read.SeizureLabelReader(split=train_split, return_tse_data=True, n_process=n_process)
        self.train_valid_slr.verbosity = 1000
        self.test_slr = read.SeizureLabelReader(split=test_split, return_tse_data=True, n_process=n_process)
        self.test_slr.verbosity = 1000
        self.test_labeling = self.test_slr[:]
        self.test_files = self.test_slr.sampleInfo
        self.test_files = [self.test_files[i].token_file_path for i in range(len(self.test_files))]
        self.valid_size = valid_size
        self.train_valid_labels = None
        self.pre_cooldown = pre_cooldown
        self.post_cooldown = post_cooldown
        self.sample_time = sample_time
        self.num_seconds=num_seconds

        #setup stuff
        if self.train_valid_labels is None:
            self.train_valid_labels = self.train_valid_slr[:]
        train_valid_files = self.train_valid_slr.sampleInfo
        train_valid_files = [train_valid_files[i].token_file_path for i in range(len(train_valid_files))]
        self.train_valid_files = train_valid_files
        patients = []
        for filename in train_valid_files:
            patients.append(read.parse_edf_token_path_structure(filename)[1])
        trainPatients, testPatients = train_test_split(list(set(patients)), test_size=self.valid_size)
        self.train_files = []
        self.valid_files = []
        self.train_labeling = []
        self.valid_labeling = []
        if use_rolling:
            self.func_call = generate_label_rolling_window
        else:
            self.func_call = seizure_series_annotate_times
        for i, file in enumerate(train_valid_files):
            if patients[i] in trainPatients:
                self.train_files.append(file)
                self.train_labeling.append(self.train_valid_labels[i])
            else:
                self.valid_files.append(file)
                self.valid_labeling.append(self.train_valid_labels[i])

    def custom_annotate(self, ann):
        return self.func_call(
            ann,
            num_seconds=self.num_seconds,
            pre_cooldown=self.pre_cooldown,
            post_cooldown=self.post_cooldown,
            sample_time=self.sample_time)



    def get_train_valid_split(self):
        return [(self.train_valid_files[i], self.custom_annotate(self.train_valid_labels[i],)) for i in range(len(self.train_valid_files))]
    def get_test_split(self):
        return Utility_Custom_Annotater(self.test_files, self.test_labeling, self.custom_annotate)[:]
        return [(self.test_files[i], self.custom_annotate(self.test_labeling[i],)) for i in range(len(self.test_files))]

    def get_train_split(self):
        return Utility_Custom_Annotater(self.train_files, self.train_labeling, self.custom_annotate)[:]
        return [(self.train_files[i], self.custom_annotate(self.train_labeling[i],)) for i in range(len(self.train_files))]

    def get_valid_split(self):
        return Utility_Custom_Annotater(self.valid_files, self.valid_labeling, self.custom_annotate)[:]
        return [(self.valid_files[i], self.custom_annotate(self.valid_labeling[i],)) for i in range(len(self.valid_files))]



class EdfDatasetSegmentedSampler(util_funcs.MultiProcessingDataset):
    DETECT_MODE="DETECT"
    PREDICT_MODE="PREDICT"
    DETECT_PREDICT_MODE="BOTH"
    def __init__(
        self,
        segment_file_tuples,
        columns_to_use=util_funcs.get_common_channel_names(),
        use_numpy=True,
        lp_cutoff=1,
        hp_cutoff=50,
        order_filt=5,
        mode=DETECT_MODE,
        resample=pd.Timedelta(seconds=constants.COMMON_DELTA),
        # num_splits_per_sample= None,
        gap = pd.Timedelta(seconds=1),
        num_samples=None,
        max_bckg_samps_per_file=None,
        n_process=4
    ):
        self.mode = mode
        self.n_process = n_process
        self.resample = resample
        self.segment_file_tuples = segment_file_tuples
        self.columns_to_use = columns_to_use
        self.use_numpy = use_numpy
        self.lp_cutoff = lp_cutoff
        self.hp_cutoff = hp_cutoff
        self.order_filt = order_filt
        self.sampleInfo = Dict()
        self.gap = gap
        self.num_samples = num_samples
        # self.num_splits_per_sample = num_splits_per_sample
        currentIndex = 0
        for token_file_path, segment in self.segment_file_tuples:
            segment = segment.reindex(np.random.permutation(segment.index)) #randomly sample from each eeg file
            num_bckg_samps_per_file = 0
            for time_period, label in segment.iteritems():
                # segment = segment.resample(gap).mode() #if gap isn't correct size, just resample
                if num_samples is not None and currentIndex >= self.num_samples:
                    break
                if max_bckg_samps_per_file is not None and num_bckg_samps_per_file >= max_bckg_samps_per_file and label == "bckg":
                    continue
                if (label != "bckg" and "sz" not in label and self.mode == EdfDatasetSegmentedSampler.DETECT_MODE):
                    continue #go to next, too close to seizure to be safe

                # for split_num in range(num_splits_per_sample):
                if self.mode == EdfDatasetSegmentedSampler.DETECT_MODE:
                    self.sampleInfo[currentIndex].label = ("sz" in label)


                if (label != "bckg" and label != "sample" and self.mode == EdfDatasetSegmentedSampler.PREDICT_MODE):
                    continue #go to next, too close to seizure to be safe or is seizure, we don't want to deal with this
                if self.mode == EdfDatasetSegmentedSampler.PREDICT_MODE:
                    self.sampleInfo[currentIndex].label = (label == "sample")

                if label == "bckg":
                    num_bckg_samps_per_file += 1
                self.sampleInfo[currentIndex].token_file_path = token_file_path
                self.sampleInfo[currentIndex].sample_num = (time_period ) / self.gap
                self.sampleInfo[currentIndex].sample_width = self.gap
                currentIndex += 1


    def __len__(self):
        return len(self.sampleInfo)


    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        indexData = self.sampleInfo[i]
        data = read.edf_eeg_2_df(indexData.token_file_path,
                                 resample=self.resample,
                                 start=pd.Timedelta(indexData.sample_num * self.gap),
                                 max_length=self.gap)

        data = data.loc[pd.Timedelta(seconds=0):self.gap].iloc[0:-1]

        data = data[self.columns_to_use]
        data = data.apply(
            lambda col: filters.butter_bandpass_filter(
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
        return data, indexData.label


def generate_label_rolling_window(ann, pre_cooldown=5, post_cooldown=None, sample_time=5, num_seconds=1):
    if post_cooldown is None:
        post_cooldown = pre_cooldown
    partial_expand = read.expand_tse_file(ann, fully_expand=False)
    timeInd = pd.timedelta_range(freq=pd.Timedelta(seconds=num_seconds), start=0, periods=ann.end.max())
    label_arr = pd.Series(index=timeInd, dtype=str).fillna("bckg")
    def is_seiz_class(label):
        if type(label) == pd.Series:
            return label.apply(lambda data: "sz" in data)
        return "sz" in label
    for i, time_index in enumerate(partial_expand.index):
        if is_seiz_class(partial_expand[i]):
            label_arr[time_index] = partial_expand[i]
        elif i != 0 and is_seiz_class(partial_expand[partial_expand.index[max(0,i-post_cooldown):i-1]]).any():
            label_arr[time_index] = "postsz" # we had seizure in past, we can't use the following period for seizure prediction
        elif is_seiz_class(partial_expand.iloc[i:i+pre_cooldown]).any():
            label_arr[time_index] = "presz" #cooldown period before seizure starts, we can't use this
        elif is_seiz_class(partial_expand.iloc[i+pre_cooldown:i+pre_cooldown+sample_time]).any():
            label_arr[time_index] = "sample" #seizure target
    return label_arr

def seizure_series_annotate_times(raw_ann,
                                  pre_cooldown=5,
                                  post_cooldown=None,
                                  sample_time=5,
                                  num_seconds=1
                                  ):
    if post_cooldown is None:
        post_cooldown = pre_cooldown
    possibleSeizureTrainTimes = []
    possibleNonseizureTrainTimes = []
    start_min = raw_ann.start.min()
    end_max = raw_ann.end.max()
    cooldown_times = []
    seizure_times = []
    preseizure_cooldown_times = []
    postseizure_cooldown_times = []
    possible_sample_times = []
    timeInd = pd.timedelta_range(freq=pd.Timedelta(seconds=num_seconds), start=0, periods=raw_ann.end.max())
    labelTimeSeries = pd.Series(index=timeInd)

    preseizure_predict_times = []
    for i, label in raw_ann.iterrows():
        if "sz" in label["label"].lower():
            seizure_times.append((label.start, label.end, label.label))
            preseizure_cooldown_times.append((max(0, label.start-pre_cooldown), label.start, label.label))
            possible_sample_times.append((max(0, label.start-pre_cooldown - sample_time), max(0, label.start-pre_cooldown), label.label))
            postseizure_cooldown_times.append((label.end, min(max(raw_ann.end), label.end+post_cooldown), label.label))
    postseizure_cooldown_times = pd.DataFrame(postseizure_cooldown_times, columns=["start", "end", "label"])
    postseizure_cooldown_times = postseizure_cooldown_times.loc[postseizure_cooldown_times["start"] != postseizure_cooldown_times["end"]]


    preseizure_cooldown_times = pd.DataFrame(preseizure_cooldown_times, columns=["start", "end", "label"])
    preseizure_cooldown_times = preseizure_cooldown_times.loc[preseizure_cooldown_times["start"] != preseizure_cooldown_times["end"]]

    seizure_times = pd.DataFrame(seizure_times, columns=["start", "end", "label"])
    seizure_times = seizure_times.loc[seizure_times["start"] != seizure_times["end"]]

    possible_sample_times = pd.DataFrame(possible_sample_times, columns=["start", "end", "label"])
    possible_sample_times = possible_sample_times.loc[possible_sample_times["start"] != possible_sample_times["end"]]
    new_ann = raw_ann.copy()

    labelTimeSeries = labelTimeSeries.fillna("bckg")
    for i, samp_time in possible_sample_times.iterrows():
        labelTimeSeries[pd.Timedelta(seconds=samp_time.start):pd.Timedelta(seconds=samp_time.end)] = "sample"
    for i, cooldown in preseizure_cooldown_times.iterrows():
        labelTimeSeries[pd.Timedelta(seconds=cooldown.start):pd.Timedelta(seconds=cooldown.end)] = "presz"
    for i, cooldown in postseizure_cooldown_times.iterrows():
        labelTimeSeries[pd.Timedelta(seconds=cooldown.start):pd.Timedelta(seconds=cooldown.end)] = "postsz"

    for i, seizure in seizure_times.iterrows():
        labelTimeSeries[pd.Timedelta(seconds=seizure.start):pd.Timedelta(seconds=seizure.end)] = seizure.label
    return labelTimeSeries

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
            lp_cutoff=1,
            hp_cutoff=50, #get close to nyq without actually hitting it to avoid errors
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
                lambda col: filters.butter_bandpass_filter(
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
