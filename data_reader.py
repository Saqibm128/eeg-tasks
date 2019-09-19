import pandas as pd
import numpy as np
import itertools
import pyedflib
from os import path
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

class EdfStandardScaler(util_funcs.MultiProcessingDataset):
    """
    Standardizes using the z-score among all the data
    """
    def __init__(self, dataset, use_only_instance_axis=True, dataset_includes_label=True, n_process=8):
        """ creates an EdfStandardScaler transformer

        Parameters
        ----------
        dataset : keras.util.Sequence-like
            that returns np.array or pd.DataFrame of some shape
        use_only_instance_axis : bool
            to standardize along all axis of each instance data, but NOT BETWEEN
        dataset_includes_label : bool
            if dataset[i] returns data, label or just data

        """
        self.dataset = dataset
        if not use_only_instance_axis:
            raise NotImplemented()
        self.use_only_instance_axis = use_only_instance_axis
        self.n_process = n_process
        self.dataset_includes_label = dataset_includes_label
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        if self.dataset_includes_label:
            data, label = self.dataset[i]
        else:
            data = self.dataset[i]

        if self.use_only_instance_axis:
            data = (data - data.mean())/data.std()

        if self.dataset_includes_label:
            return data, label
        else:
            return data

class SeizureLabelReader(util_funcs.MultiProcessingDataset):
    def __init__(self, split=None, ref="01_tcp_ar", return_tse_data=False, is_present_only=True, edf_token_paths=[], sampleInfo=None, n_process=4, overwrite_sample_info_label=True):
        """ Provides access to an array-like that can create labels matching sampleInfo
        or if edf_token_paths is available

        Parameters
        ----------
        is_present_only : bool
            Whether to use simple task of whether seizure occurred or to use another mode
        sampleInfo : addict.Dict
            optional dict if using the random ensemble,
            in form of EdfDatasetEnsembler.sampleInfo
            (has fileTokenPath, sampleNum, max_length)
        overwrite_sample_info_label : bool
            Whether to overwrite the label info in the sampleInfo # DEBUG: ict passed in

        Returns
        -------
        SeizureLabelReader
            array-like to access the label info
        """
        if not is_present_only:
            raise NotImplementedError("TODO: maybe allow ways to get labels over time or if seizure is about to occur")
        self.sampleInfo = sampleInfo
        if sampleInfo is None:
            token_files = get_all_token_file_names(split, ref)
            self.sampleInfo = Dict()
            for i, token_file in enumerate(token_files):
                self.sampleInfo[i].token_file_path = token_file
        self.is_present_only  = is_present_only
        self.n_process = n_process
        self.verbosity = 100
        self.edf_token_paths = edf_token_paths
        self.overwrite_sample_info_label = overwrite_sample_info_label
        self.return_tse_data = return_tse_data

    def self_assign_to_sample_info(self, convert_to_int):
        labels = self[:]
        for i in range(len(self.sampleInfo)):
            self.sampleInfo[i].label = labels[i]
            if convert_to_int:
                self.sampleInfo[i].label = int(labels[i])


    def __len__(self):
        if self.sampleInfo is not None:
            return len(self.sampleInfo)
        else:
            return len(self.edf_token_paths)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        if self.is_present_only and self.sampleInfo is not None:
            token_file_path = self.sampleInfo[i].token_file_path
            sampleNum = self.sampleInfo[i].sample_num
            sample_width = self.sampleInfo[i].sample_width
            label_file = convert_edf_path_to_tse(token_file_path)
            seiz_label = read_tse_file(label_file)
            if self.return_tse_data:
                return seiz_label
            startTime = pd.Timedelta(sampleNum * sample_width).seconds
            endTime = pd.Timedelta(sampleNum*sample_width + sample_width).seconds
            #look for where the slice lands
            seizInfoSlice = seiz_label.loc[(seiz_label.end > startTime) & (seiz_label.end <= endTime).shift(1)]
            label = (seizInfoSlice.label != "bckg").any()

            if self.overwrite_sample_info_label: #only works if n_process is 1???
                self.sampleInfo[i].label = label

            return label





class EdfFFTDatasetTransformer(util_funcs.MultiProcessingDataset):
    freq_bins = [0.2 * i for i in range(50)] + list(range(10, 80, 1)) #default freq bins unless if you override this
    """Implements an indexable dataset applying fft to entire timeseries,
        returning histogram bins of fft frequencies

    Parameters
    ----------
    edf_dataset : EdfDataset
        an array-like returning the raw channel data and the output as a tuple or a single result
    freq_bins : type
        Description of parameter `freq_bins`.
    n_process : type
        Description of parameter `n_process`.
    precache : type
        Description of parameter `precache`.
    window_size : type
        Description of parameter `window_size`.
    non_overlapping : type
        Description of parameter `non_overlapping`.

    """

    def __init__(
        self,
        edf_dataset,
        is_tuple_data=True,
        is_pandas_data=True,
        freq_bins=freq_bins,
        n_process=None,
        precache=False,
        window_size=None,
        non_overlapping=True,
        return_ann=True,
        return_numpy=False #return pandas.dataframe if possible (if windows_size is false)
    ):
        """Used to read the raw data in

        Parameters
        ----------
        edf_dataset : EdfDataset
            Array-like returning the channel data (channel by time) and annotations (doesn't matter what the shape is)
        freq_bins : array
            Used to segment the frequencies into histogram bins
        n_process : int
            Used to define the number of processes to use for large reads in. If None, uses cpu count
        precache : bool
            Use to load all data at beginning and keep cache of it during operations
        window_size : pd.Timedelta
            If None, runs the FFT on the entire datset. If set, uses overlapping windows to run fft on
        non_overlapping : bool
            If true, the windows are used to reduce dim red, we don't use rolling-like behavior
        return_ann : bool
            If false, we just output the raw data
        Returns
        -------
        None

        """
        self.return_numpy = return_numpy
        if not is_tuple_data:
            return_ann = False #you can't return annotation data if annotation isn't included
        self.is_tuple_data = is_tuple_data
        self.is_pandas_data = is_pandas_data
        self.edf_dataset = edf_dataset
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.precache = False
        self.freq_bins = freq_bins
        self.window_size = window_size
        self.non_overlapping = non_overlapping
        self.return_ann = return_ann
        if precache:
            print(
                "starting precache job with: {} processes".format(
                    self.n_process))
            self.data = self[:]
        self.precache = precache

    def __len__(self):
        return len(self.edf_dataset)

    def __getitem__(self, i):
        if self.precache:
            return self.data[i]
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        if self.window_size is None:
            original_data_label = self.edf_dataset[i]
            if self.is_tuple_data:
                original_data, label = original_data_label
            else:
                original_data = original_data_label
            if self.is_pandas_data:
                columns = original_data.columns
                original_data = original_data.values
            else:
                columns = list(range(original_data.shape[1]))
            fft_data = np.nan_to_num(
                np.abs(
                    np.fft.fft(
                        original_data,
                        axis=0)))
            fft_freq = np.fft.fftfreq(fft_data.shape[0], d=constants.COMMON_DELTA)
            fft_freq_bins = self.freq_bins
            new_fft_hist = pd.DataFrame(
                index=fft_freq_bins[:-1], columns=columns)
            for i, name in enumerate(columns):
                new_fft_hist[name] = np.histogram(
                    fft_freq, bins=fft_freq_bins, weights=fft_data[:, i])[0]
            if self.return_numpy:
                new_fft_hist = new_fft_hist.values
            if not self.return_ann:
                return new_fft_hist
            return new_fft_hist, label
        else:
            window_count_size = int(
                self.window_size /
                pd.Timedelta(
                    seconds=constants.COMMON_DELTA))

            original_data_label = self.edf_dataset[i]
            if self.is_tuple_data:
                original_data, label = original_data_label
            else:
                original_data = original_data_label
            fft_data = original_data.values
            fft_data_windows = np_rolling_window(
                np.array(fft_data.T), window_count_size)
            if self.non_overlapping:
                fft_data_windows = fft_data_windows[:, list(
                    range(0, fft_data_windows.shape[1], window_count_size))]
            fft_data = np.abs(
                np.fft.fft(
                    fft_data_windows,
                    axis=2))  # channel, window num, frequencies
            fft_freq_bins = self.freq_bins
            new_hist_bins = np.zeros(
                (fft_data.shape[0], fft_data.shape[1], len(fft_freq_bins) - 1))
            fft_freq = np.fft.fftfreq(window_count_size, d=constants.COMMON_DELTA)
            for i, channel in enumerate(fft_data):
                for j, window_channel in enumerate(channel):
                    new_hist_bins[i, j, :] = np.histogram(
                        fft_freq, bins=fft_freq_bins, weights=window_channel)[0]
            if not self.return_ann:
                return new_hist_bins
            if (self.edf_dataset.expand_tse and not self.non_overlapping):
                return new_hist_bins, label.rolling(window_count_size).mean(
                )[:-window_count_size + 1].fillna(method="ffill").fillna(method="bfill")
            elif (self.edf_dataset.expand_tse and self.non_overlapping):
                annotations = label.rolling(window_count_size).mean()[
                    :-window_count_size + 1]
                return new_hist_bins, annotations.iloc[list(range(
                    0, annotations.shape[0], window_count_size))].fillna(method="ffill").fillna(method="bfill")
            else:
                return new_hist_bins, label


class EdfDataset(util_funcs.MultiProcessingDataset):
    """Basic access to the raw data. Is the first layer in any/all data processing
    and is usually what is passed to the other datasets/transformers

    TODO: possibly create a parallel backend so that same calls to same data input
    with multiple higher level layers don't do same expensive io/mem allocations
    in this base dataset

    Parameters
    ----------
    data_split : type
        Description of parameter `data_split`.
    ref : type
        Description of parameter `ref`.
    resample : type
        Description of parameter `resample`.

    Attributes
    ----------
    manager : multiprocessing.Manager
        used to manage multiprocessing. TODO: not implemented
    edf_tokens : list
        a list of edf file paths to consider, assumes a corresponding tse file
        exists
    n_process : int
        When indexing by slice, use multiprocessing to speed up execution
    data_split : str
    ref : str
    resample : pd.Timedelta

    """

    def __init__(
            self,
            data_split,
            ref,
            num_files=None,
            resample=pd.Timedelta(
                seconds=constants.COMMON_DELTA),
            start_offset=pd.Timedelta(seconds=0), #start at 0 unless if we want something different
            max_length=None,
            expand_tse=False, #Save memory, don't try to make time by annotation df
            dtype=np.float32,
            n_process=None,
            use_average_ref_names=True,
            filter=True,
            lp_cutoff=1,
            hp_cutoff=50, #get close to nyq without actually hitting it
            order_filt=5,
            columns_to_use=util_funcs.get_common_channel_names(),
            use_numpy=False
            ):
        self.data_split = data_split
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.ref = ref
        self.resample = resample
        self.dtype = dtype
        self.start_offset = start_offset
        self.max_length = max_length
        self.manager = mp.Manager()
        self.edf_tokens = get_all_token_file_names(data_split, ref)
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

    def __len__(self):
        return len(self.edf_tokens)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        data, ann = get_edf_data_and_label_ts_format(
            self.edf_tokens[i], resample=self.resample, expand_tse=self.expand_tse, dtype=self.dtype, start=self.start_offset, max_length=self.max_length)
        if (self.max_length != None and max(data.index) > self.max_length):
            if type(self.max_length) == pd.Timedelta:
                data = data.loc[pd.Timedelta(seconds=0):self.max_length]
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
        return data, ann

def parse_edf_token_path_structure(edf_token_path):
    remaining, token = path.split(edf_token_path)
    remaining, session = path.split(remaining)
    remaining, patient = path.split(remaining)
    remaining, patient_prefix = path.split(remaining) #first 3 digits of patient id
    remaining, split = path.split(remaining)
    return split, patient, session, token



def get_edf_data_and_label_ts_format(
    edf_path, expand_tse=True, resample=pd.Timedelta(
        seconds=constants.COMMON_DELTA), start=pd.Timedelta(seconds=0), dtype=np.float32, max_length=None):
    try:
        edf_data = edf_eeg_2_df(edf_path, resample, dtype=dtype, start=start, max_length=max_length)
        tse_data_path = convert_edf_path_to_tse(edf_path)
        if expand_tse:
            tse_data_ts = read_tse_file_and_return_ts(
                tse_data_path, edf_data.index)
        else:
            tse_data_ts = read_tse_file(tse_data_path)
    except Exception as e:
        print("could not read: {}".format(edf_path))
        print(e)
        raise e
    return edf_data, tse_data_ts


def read_tse_file(tse_path):
    tse_data_lines = []
    with open(tse_path, 'r') as f:
        for line in f:
            if "#" in line:
                continue  # this is a comment
            elif "version" in line:
                assert "tse_v1.0.0" in line  # assert file is correct version
            elif len(line.strip()) == 0:
                continue  # Just blank space, continue
            else:
                line = line.strip()
                subparts = line.split()
                tse_data_line = pd.Series(index=['start', 'end', 'label', 'p'])
                tse_data_line['start'] = float(subparts[0])
                tse_data_line['end'] = float(subparts[1])
                tse_data_line['label'] = str(subparts[2])
                tse_data_line['p'] = float(subparts[3])
                tse_data_lines.append(tse_data_line)
    tse_data = pd.concat(tse_data_lines, axis=1).T
    return tse_data


def convert_edf_path_to_tse(edf_path):
    return edf_path[:-4] + ".tse"


def convert_edf_path_to_txt(edf_path):
    return edf_path[:-9] + ".txt"


def get_all_clinical_notes(session_path, edf_convert=True):
    """ gets the freeform text

    Parameters
    ----------
    path : string
        String to the file
    edf_convert : bool
        If this is actually a edf file passed in, we convert to txt file

    Returns
    -------
    str
        raw clinical notes
    """
    if edf_convert:
        clinical_notes_path = convert_edf_path_to_txt(session_path)
    else:
        clinical_notes_path = session_path
    with open(clinical_notes_path, 'rb') as f:
        lines = f.readlines()
    res = ""
    for line in lines:
        res += str(line)
    return res


def read_tse_file_and_return_ts(tse_path, ts_index):
    ann_y = read_tse_file(tse_path)

    return expand_tse_file(ann_y, ts_index)


def expand_tse_file(ann_y, ts_index, dtype=np.float32):
    ann_y_t = pd.DataFrame(columns=get_annotation_types(), index=ts_index)
    ann_y.apply(lambda row: ann_y_t[row['label']].loc[pd.Timedelta(
        seconds=row['start']):pd.Timedelta(seconds=row['end'])].fillna(row['p'], inplace=True), axis=1)
    ann_y_t.fillna(0, inplace=True)
    return ann_y_t


def edf_eeg_2_df(path, resample=None, dtype=np.float32, start=0, max_length=None):
    """ Transforms from EDF to pd.df, with channel labels as columns.
        This does not attempt to concatenate multiple time series but only takes
        a single edf filepath

    Parameters
    ----------
    path : str
        path of the edf file

    resample : pd.Timedelta
        if None, returns original data with original sampling
        otherwise, resamples to correct Timedelta using forward filling

    dtype : dtype
        used to reduce memory consumption (np.float64 can be expensive)

    start : int or pd.Timedelta
        which place to start at

    Returns
    -------
    pd.DataFrame
        index is time, columns is waveform channel label

    """
    with pyedflib.EdfReader(path, check_file_size=pyedflib.DO_NOT_CHECK_FILE_SIZE) as reader:
        channel_names = [headerDict['label']
                         for headerDict in reader.getSignalHeaders()]
        sample_rates = [headerDict['sample_rate']
                        for headerDict in reader.getSignalHeaders()]
        start_time = pd.Timestamp(reader.getStartdatetime())
        all_channels = []
        for i, channel_name in enumerate(channel_names):
            if type(start) == pd.Timedelta: #we ask for time t=1 s, then we take into account sample rate
                start_count_native_freq = start/pd.Timedelta(seconds=1/sample_rates[i])
            else:
                start_count_native_freq = start
            if max_length is None: #read everything
                signal_data = reader.readSignal(i, start=start_count_native_freq)
            else:
                numStepsToRead = int(np.ceil(max_length / pd.Timedelta(seconds=1/sample_rates[i]))) + 5 #adding a fudge factor of 5 cuz y not.
                signal_data = reader.readSignal(i, start=start_count_native_freq, n=numStepsToRead)

            signal_data = pd.Series(
                signal_data,
                index=pd.date_range(
                    start=start_time,
                    freq=pd.Timedelta(seconds=1 / sample_rates[i]),
                    periods=len(signal_data)
                ),
                name=channel_name
            )
            all_channels.append(signal_data)
    data = pd.concat(all_channels, axis=1)
    data.index = data.index - data.index[0]
    data = data.astype(dtype)
    if resample is not None:
        data = data.resample(resample).mean()
    return data


def get_patient_dir_names(data_split, ref, full_path=True):
    """ Gets the path names of the folders holding the patient info

    Parameters
    ----------
    data_split : str
        Which data split the patient belongs to
    ref : str
        which reference voltage system that is used
    full_path : bool
        Whether or not to return abs path

    Returns
    -------
    list
        list of strings describing path

    """
    assert data_split in get_data_split()
    assert ref in get_reference_node_types()
    config = read_config()
    if data_split is None:
        root_dir_path = config["data_dir_root"] + "/" + ref
    else:
        root_dir_entry = data_split + "_" + ref
        root_dir_path = config[root_dir_entry]
    subdirs = get_abs_files(root_dir_path)
    patient_dirs = list(itertools.chain.from_iterable(
        [get_abs_files(subdir) for subdir in subdirs]))
    if full_path:
        return patient_dirs
    else:
        return [path.basename(patient_dir) for patient_dir in patient_dirs]


def get_session_dir_names(data_split, ref, full_path=True, patient_dirs=None):
    """Gets the path names of the folders holding the session info
        (patients can have multiple eeg sessions)

    Parameters
    ----------
    data_split : str
        Which data split the patient belongs to
    ref : str
        which reference voltage system that is used
    full_path : bool
        Whether or not to return abs path

    Returns
    -------
    list
        list of strings describing path
    """
    assert data_split in get_data_split()
    assert ref in get_reference_node_types()
    if patient_dirs is None:
        patient_dirs = get_patient_dir_names(data_split, ref)
    session_dirs = list(itertools.chain.from_iterable(
        [get_abs_files(patient_dir) for patient_dir in patient_dirs]))
    if full_path:
        return session_dirs
    else:
        return [path.basename(session_dir) for session_dir in session_dirs]


def get_all_token_file_names(data_split, ref, full_path=True):
    session_dirs = get_session_dir_names(data_split, ref)
    token_fns = list(itertools.chain.from_iterable(
        [get_token_file_names(session_dir) for session_dir in session_dirs]))
    if full_path:
        return token_fns
    else:
        return [path.basename(token_fn) for token_fn in token_fns]


def get_token_file_names(session_dir_path, full_path=True):
    sess_file_names = get_abs_files(session_dir_path)
    time_series_fns = [fn for fn in sess_file_names if fn[-4:] == '.edf']
    if full_path:
        return time_series_fns
    else:
        return [path.basename(fn) for fn in time_series_fns]


def get_session_data(session_dir_path):
    time_series_fns = get_token_file_names(session_dir_path)
    signal_dfs = []
    for fn in time_series_fns:
        signal_dfs.append(edf_eeg_2_df(fn))
    return pd.concat(signal_dfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Holds utility functions for reading data. As a script, stores a copy of the fft dataset as pkl format')
    parser.add_argument("data_split", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument(
        "--path",
        type=str,
        default="",
        description="directory to store output file in")
    parser.add_argument("--num_files", type=int, default=None)
    # not a real soft-run but oh well
    parser.add_argument("--dry-run", action="store_true")
    # use s2s data to make a cached pickle instead
    parser.add_argument("--use_s2s", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.use_s2s:
        edf_dataset = EdfFFTDatasetTransformer(
            EdfDataset(
                args.data_split,
                args.ref,
                num_files=args.num_files,
                expand_tse=False),
            precache=True)
        pkl.dump(
            edf_dataset.data,
            open(
                args.path +
                "{}_{}{}_fft.pkl".format(
                    args.data_split,
                    args.ref,
                    "" if args.num_files is None else "_n_{}".format(
                        args.num_files)),
                'wb'))
    elif not args.dry_run and args.use_s2s:
        edf_dataset = EdfFFTDatasetTransformer(
            EdfDataset(
                args.data_split,
                args.ref,
                num_files=args.num_files,
                expand_tse=False),
            window_size=pd.Timedelta(
                seconds=10),
            non_overlapping=True)
        s2s_dataset = Seq2SeqFFTDataset(edfFFTData=edf_dataset, n_process=12)
        token_fns = edf_dataset.edf_dataset.edf_tokens
        pkl.dump(
            (token_fns,
             s2s_dataset[:]),
            open(
                args.path +
                "s2s_{}_{}{}_fft.pkl".format(
                    args.data_split,
                    args.ref,
                    "" if args.num_files is None else "_n_{}".format(
                        args.num_files)),
                'wb'))

    else:
        print("Dry-Run, checking all EDF files are readable")
        token_files = get_all_token_file_names(args.data_split, args.ref)
        if args.num_files is not None:
            token_files = token_files[:args.num_files]

        times = []
        for path in token_files:
            try:
                with pyedflib.EdfReader(path, check_file_size=pyedflib.DO_NOT_CHECK_FILE_SIZE) as reader:
                    times.append(
                        reader.readSignal(0).shape[0] /
                        reader.getSignalHeader(0)["sample_rate"])
            except BaseException:
                print("Path: {} is unsuccessful".format(path))
        pd.Series(times).to_csv("times.csv")
