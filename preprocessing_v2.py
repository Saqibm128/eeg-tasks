#uses prep pipeline
import pyprep
import constants

#map edf channel names to MNE_CHANNEL_EDF_MAPPING
def edf_chn_to_mne(data):
    return [constants.MNE_CHANNEL_EDF_MAPPING[chn] if chn in constants.MNE_CHANNEL_EDF_MAPPING else chn for chn in data.ch_names]


class EDFSubset(util_funcs.MultiProcessingDataset):
    def __init__(self, max_sec_size=10, output_dir="/n/scratch2/ms994/pyprepOutput"):
        self.max_sec_size = max_sec_size
        self.output
