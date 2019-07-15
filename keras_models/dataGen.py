# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import keras
import pandas as pd

#Wrapper classes for batch training in Keras

def three_dim_pad(data, mask_value, num_channels=1, max_length=None):
    """Used to pad in a list of variable length data

    Parameters
    ----------
    data : list
        List of data in shape time by features.
    mask_value : float or int
        used as placeholder for masking layer in an LSTM

    Returns
    -------
    np.array
        correctly sized np.array for batch with mask values filled in

    """
    # for n_batch, n_timestep, n_input matrix, pad_sequences fails
    lengths = [datum.shape[0] for datum in data]
    if max_length is not None:
        maxLength = max(lengths)
    paddedBatch = np.zeros((len(data), maxLength, *data[0].shape[1:], num_channels))
    paddedBatch.fill(mask_value)
    for i, datum in enumerate(data):
        if type(datum) == pd.DataFrame:
            datum = datum.values
        if num_channels == 1:
            datum = datum.reshape(*datum.shape, 1)
        if max_length is None:
            paddedBatch[i, 0:lengths[i], :] = datum
        else:
            datum = datum[0:max_length,:]
            paddedBatch[i, 0:datum.shape[0], :] = datum
    return paddedBatch

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_x_y(self, id):
        return np.load('data/' + ID + '.npy'), self.labels[ID]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,], y[i] = self.get_x_y(ID)




class EdfDataGenerator(DataGenerator):
    'Can accept EdfDataset and any of its intermediates to make data (i.e. sftft)'
    def __init__(self, dataset, mask_value=-10000, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, max_length=None):
        super().__init__(list_IDs=list(range(len(dataset))), labels=labels, batch_size=batch_size, dim=dim, n_channels=n_channels,
                     n_classes=n_classes, shuffle=shuffle)
        self.dataset = dataset
        self.mask_value=mask_value
        self.max_length=max_length
    def get_x_y(self, i):
        if self.labels is not None:
            y = self.labels[i]
        data = self.dataset[i]
        x = [datum[0] for datum in data]
        if self.labels is None:
            y = [datum[1] for datum in data]

        return x, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        '''Overriding this to deal with None dimensions (i.e. variable length times)
        and allow for MultiProcessingDataset to work (only works for slices) '''

        x, y = self.get_x_y(list_IDs_temp)
        x = three_dim_pad(x, self.mask_value, max_length=self.max_length)
        if self.labels is None:
            return x, y
        else:
            return x,  keras.utils.to_categorical(y, num_classes=self.n_classes)


        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)