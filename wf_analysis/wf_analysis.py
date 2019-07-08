import util_funcs

class BandPassTransformer(util_funcs.MultiProcessingDataset):
    def __init__(self, edfRawData, n_process=None, bandpass_gaps=[]):
        self.edfRawData = edfRawData
        self.n_process = n_process
        self.bandpass_gaps = bandpass_gaps
    def __len__(self):
        return len(self.edfRawData)
    def __getitem__(self, i):
        if type(i) == slice:
            return self.getItemSlice(i)
        rawData, ann = self.edfRawData[i]
