import util_funcs
import filters
import pywt
import tsfresh.feature_extraction.feature_calculators as feats
import constants


def norm_num_peaks_func(n):
    return lambda x: feats.number_peaks(x, n) / len(x)


def norm_num_vals_func(n):
    return lambda x: feats.number_peaks(-x, n) / len(x)


def autocorrelation(lag):
    return lambda x: feats.autocorrelation(x, lag)


class BandPassTransformer(util_funcs.MultiProcessingDataset):
    """Transforms a set of channel data intoo segmented signals based on a
    bandpass filter. Used primarily to separate alpha, beta, gamma, and delta components

    Parameters
    ----------
    edfRawData : type
        Description of parameter `edfRawData`.
    n_process : type
        Description of parameter `n_process`.
    bandpass_gaps : type
        Description of parameter `bandpass_gaps`.

    Attributes
    ----------
    edfRawData
    n_process
    bandpass_gaps

    """

    def __init__(self, edfRawData, n_process=None, bandpass_freqs=[]):
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

    def __len__(self):
        return len(self.edfRawData)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.getItemSlice(i)
        rawData, ann = self.edfRawData[i]
        bandPassColumns = [
            rawDataColumn +
            str(freq) for rawDataColumn in rawData.columns for freq in self.bandpass_freqs]
        newBandPass = pd.DataFrame(columns=[bandPassColumns])
        for rawDataColumn in rawData.columns:
            for freqs in self.bandpass_freqs:
                lp, hp = freqs
                singChannelData = rawData[rawDataColumn]
                filters.butter_bandpass_filter(
                    singChannelData, lp, hp, constants.fs)
