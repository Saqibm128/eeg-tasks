import pandas as pd
import numpy as np
import util_funcs
import wf_analysis.filters as filters
import pywt
import tsfresh.feature_extraction.feature_calculators as feats
import constants
from scipy.signal import coherence
import pywt
import multiprocessing as mp


def norm_num_peaks_func(n):
    return lambda x: feats.number_peaks(x, n) / len(x)


def norm_num_vals_func(n):
    return lambda x: feats.number_peaks(-x, n) / len(x)


def autocorrelation(lag):
    return lambda x: feats.autocorrelation(x, lag)

class ConcatenationDataset(util_funcs.MultiProcessingDataset):
    '''
    Avoid a possible bottleneck where we try to call the base segments too much, just concatenate a ton of transforms together
    '''
    def __init__(self, datasets, flatten=True):
        self.datasets = datasets
        self.flatten = flatten

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        x = []
        y = None
        for dataset in self.datasets:
            xy = dataset[i]
            if type(xy) == tuple:
                if not self.flatten:
                    x.append(xy[0])
                else:
                    if type(xy[0]) == pd.DataFrame:
                        x.append(xy[0].values.flatten())
                    elif type(xy[0]) == pd.Series:
                        x.append(xy[0].values.flatten())
                    else:
                        x.append(xy[0].flatten())
                if y == None:
                    y = xy[1]
            else:
                if type(xy) == pd.DataFrame or type(xy) == pd.Series:
                    xy = xy.values.flatten()
                else:
                    xy = xy.flatten()
                x.append(xy)
        if self.flatten:
            x = np.hstack(x)
        return x, y

class SimpleHandEngineeredDataset(util_funcs.MultiProcessingDataset):
    def __init__(self, edfRawData, n_process=None, features = [], f_names = [], max_size=None, vectorize=None, is_pandas_data=True):
        assert len(features) == len(f_names)
        self.edfRawData = edfRawData
        self.n_process = n_process
        if n_process is None:
            self.n_process = mp.cpu_count()
        self.features = features
        self.f_names = f_names
        self.max_size = max_size
        self.vectorize = vectorize
        self.is_pandas_data = is_pandas_data

    def __len__(self):
        return len(self.edfRawData)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            print("starting simple handEngineeredData")
            return self.getItemSlice(i)
        rawData, ann = self.edfRawData[i]
        if self.max_size is not None and max(rawData.index) < self.max_size:
            rawData = rawData[:self.max_size]
        if not self.is_pandas_data:
            rawData = pd.DataFrame(rawData)
        handEngineeredData = pd.DataFrame(index=rawData.columns, columns=self.f_names)

        for i, feature in enumerate(self.features):
            handEngineeredData[self.f_names[i]] = rawData.apply(lambda x: feature(x.values))
        if self.vectorize == "full":
            return handEngineeredData.values.reshape(-1)
        if self.vectorize == "mean":
            return handEngineeredData.values.mean()
        return handEngineeredData

class EdfDWTDatasetTransformer(util_funcs.MultiProcessingDataset):
    def __init__(
        self,
        edf_dataset,
        n_process=None,
        precache=False,
        wavelet="db1",
        return_ann=True,
        max_coef=None,
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
        self.edf_dataset = edf_dataset
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.precache = False
        self.return_ann = return_ann
        if precache:
            print(
                "starting precache job with: {} processes".format(
                    self.n_process))
            self.data = self[:]
        self.precache = precache
        self.wavelet = wavelet
        self.max_coef = max_coef

    def __len__(self):
        return len(self.edf_dataset)

    def __getitem__(self, i):
        if self.precache:
            return self.data[i]
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        original_data = self.edf_dataset[i]
        return original_data.apply(
            lambda x: pywt.dwt(
                x.values,
                self.wavelet)[0],
            axis=0)[
            :self.max_coef]

class CoherenceTransformer(util_funcs.MultiProcessingDataset):
    def __init__(self, edfRawData, n_process=None, coherence_all=True, coherence_pairs=None, average_coherence=True, coherence_bins=None, columns_to_use=util_funcs.get_common_channel_names(), is_pandas=True, is_tuple_data=True):
        """

        Parameters
        ----------
        edfRawData : DataFrame
            An array-like holding the data for coherence
        n_process : int
            number of processes to use when indexing a slice
        coherence_all : bool
            If to do pair-wise coherence on all channels, if so we increase
            num features to n*n-1
        coherence_pairs : list
            If coherence_all is false, pass in a list of tuples holding columns
            to run coherence measurements on
        average_coherence : bool
            If true, just do an average of all coherences over all represented
            frequencies. If False, use coherence_bins to histogram bin everything

        Returns
        -------
        CoherenceTransformer
            Array-like

        """
        self.edfRawData = edfRawData
        self.n_process = n_process
        self.is_pandas = is_pandas
        self.coherence_all = coherence_all
        self.coherence_pairs = coherence_pairs
        self.average_coherence = average_coherence
        self.coherence_bins = coherence_bins
        self.columns_to_use = columns_to_use
        self.is_pandas = is_pandas
        self.is_tuple_data = is_tuple_data
    def __len__(self):
        return len(self.edfRawData)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            print("starting simple coherence")
            return self.getItemSlice(i)
        if self.is_tuple_data:
            raw_data, ann = self.edfRawData[i]
        else:
            raw_data = self.edfRawData[i]
        if self.coherence_all:
            coherence_pairs = []
            if self.is_pandas:
                for k in range(len(self.columns_to_use) - 1):
                    column_1 = self.columns_to_use[k]
                    for j in range(k + 1, len(self.columns_to_use)):
                        column_2 = self.columns_to_use[j]
                        coherence_pairs.append((column_1, column_2))
            else:
                for k in range(raw_data.shape[1] - 1):
                    for j in range(k + 1, raw_data.shape[1]):
                        coherence_pairs.append((j, k))

        else:
            coherence_pairs = self.coherence_pairs

        if self.average_coherence:
            toReturn = pd.Series()
            if self.is_pandas:
                for column_1, column_2 in coherence_pairs:
                    toReturn["coherence {}".format((column_1, column_2))] =  np.mean(coherence(raw_data[column_1], raw_data[column_2], fs=constants.COMMON_FREQ, nperseg=constants.COMMON_FREQ/4)[1])
            else:
                for column_1, column_2 in coherence_pairs:
                    toReturn["coherence {}".format((column_1, column_2))] =  np.mean(coherence(raw_data.T[column_1], raw_data.T[column_2], fs=constants.COMMON_FREQ, nperseg=constants.COMMON_FREQ/4)[1])
        else:
            raise NotImplemented("yet")
        return toReturn, ann




class BandPassTransformer(util_funcs.MultiProcessingDataset):
    """Transforms a set of channel data into segmented signals based on a
            bandpass filter. Used primarily to separate alpha, beta, gamma,
            and delta components for further feature extraction

    Parameters
    ----------
    edfRawData : DataFrame
        Description of parameter `edfRawData`.
    n_process : int
        Description of parameter `n_process`.
    bandpass_gaps : type
        Description of parameter `bandpass_gaps`.

    Attributes
    ----------
    edfRawData
    n_process
    bandpass_gaps

    """

    def __init__(self, edfRawData, n_process=None, bandpass_freqs=[], order=5):
        """

        Parameters
        ----------
        edfRawData : Array-like
            array-like that returns data in the form of a dataframe
            (channel by time) and annotation data
        n_process : int
            number of processes to use
        bandpass_freqs : list
            list of tuples, deterimining low pass and high pass frequencies

        Returns
        -------
        BandPassTransformer
            newly created array-like
        """
        self.edfRawData = edfRawData
        self.n_process = n_process
        self.bandpass_freqs = bandpass_freqs
        self.order = order

    def __len__(self):
        return len(self.edfRawData)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        rawData, ann = self.edfRawData[i]
        bandPassColumns = [
            rawDataColumn +
            str(freqs) for rawDataColumn in rawData.columns for freqs in self.bandpass_freqs]
        newBandPass = pd.DataFrame(columns=[bandPassColumns])
        for rawDataColumn in rawData.columns:
            for freqs in self.bandpass_freqs:
                lp, hp = freqs
                singChannelData = rawData[rawDataColumn]
                newBandPass[rawDataColumn + str(freqs)] = filters.butter_bandpass_filter(
                    singChannelData, lp, hp, constants.COMMON_FREQ, order=self.order)
        return newBandPass, ann
