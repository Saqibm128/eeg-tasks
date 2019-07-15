import pandas as pd
import numpy as np
import util_funcs
import constants

class BasicSpatialDataset(util_funcs.MultiProcessingDataset):
    """Class to apply naive mapping of channels to spatial image to channel data.

    Parameters
    ----------
    dataset : pd.DataFrame
        returns time data with same names as spatialMapping, and y labeling (i.e. either EdfDataset or EdfFFTDatasetTransformer)
    spatialMapping : 2d array
        a 2d array to make an image for each time point given by data, if 0 then empty 'pixel'
    n_process : int
        max processes to use

    Attributes
    ----------
    n_process
    dataset
    spatialMapping

    """
    def __init__(self, dataset, spatialMapping=constants.SIMPLE_CONV2D_MAP, n_process=8, columns_to_use=None):
        self.n_process = n_process
        self.dataset = dataset
        self.spatialMapping = spatialMapping
        if columns_to_use is None:
            self.columns_to_use = []
            for row in spatialMapping:
                for col in row:
                    if col != 0:
                        self.columns_to_use.append(col)
        else:
            self.columns_to_use = columns_to_use
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        x, y = self.dataset[i]
        spatialX = np.ndarray((x.shape[0], len(self.spatialMapping), len(self.spatialMapping[0])))
        for i,row in enumerate(self.spatialMapping):
            for j,col in enumerate(row):
                if col == 0:
                    continue
                else:
                    spatialX[:,i,j] = x[col]
        return spatialX, y
