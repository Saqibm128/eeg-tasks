# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from addict import Dict
from numpy.random import choice
import multiprocessing as mp
import time

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
    if max_length is None:
        max_length = max(lengths)
    paddedBatch = np.zeros((len(data), max_length, *data[0].shape[1:], num_channels))
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
    '''
    Generates data for Keras, based on code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Primarily made because I thought data wouldn't fit inside memory
    '''
    def __init__(self, list_IDs, labels, data=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

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
                 n_classes=10, class_type="nominal", shuffle=True, max_length=None, time_first=True, precache=False, xy_tuple_form=True, use_background_process=False):

        super().__init__(list_IDs=list(range(len(dataset))), labels=labels, batch_size=batch_size, dim=dim, n_channels=n_channels,
                     n_classes=n_classes, shuffle=shuffle)
        if not xy_tuple_form:
            self.list_IDs = list(range(len(dataset[0]))) #the dataset is a tuple of x and y. grab x and use that length.
        self.dataset = dataset
        self.mask_value=mask_value
        self.max_length=max_length
        self.time_first = time_first
        self.xy_tuple_form = xy_tuple_form
        self.class_type=class_type
        self.use_background_process=use_background_process
        if self.use_background_process:
            self.manager = mp.Manager()
            self.queue = self.manager.Queue()

        if precache: #just populate self.labels too if we are precaching anyways
            self.dataset = dataset[:]
            if self.labels is None:
                self.labels = np.array([datum[1] for datum in self.dataset])
        self.precache = precache
        if type(self.labels) == list:
            self.labels = np.array(self.labels)

    def background_population(self):
        for i in range(len(self)):
            while self.queue.full():
                time.sleep(0.1)
            self.queue.put(self.__getitem__(i, accessed_by_background=True))
            print(i)


    def start_background(self):
        #use if u want to run train_on_batch
        self.process = mp.Process(target=self.background_population)
        self.process.start()

    def create_validation_train_split(self, validation_size=0.1):
        '''
        Used to make a split in the EdfDataGenerator with respect to the labels; wasn't sure where better to place this for a data_generator class
        '''
        if not self.precache:
            raise Exception("NOT IMPLEMENTED")
        train_data, validation_data, train_labels, validation_labels = train_test_split(self.dataset, self.labels, test_size=validation_size, stratify=self.labels)

        train_data_gen = EdfDataGenerator(dataset=train_data, mask_value=self.mask_value, labels=train_labels, batch_size=self.batch_size, dim=self.dim, n_channels=self.n_channels,
                     n_classes=self.n_classes, shuffle=self.shuffle, max_length=self.max_length, time_first=self.time_first, precache=self.precache, class_type=self.class_type)


        validation_data_gen = EdfDataGenerator( dataset=validation_data, mask_value=self.mask_value, labels=validation_labels, batch_size=self.batch_size, dim=self.dim, n_channels=self.n_channels,
                     n_classes=self.n_classes, shuffle=self.shuffle, max_length=self.max_length, time_first=self.time_first, precache=self.precache, class_type=self.class_type)

        return train_data_gen, validation_data_gen


    def get_x_y(self, i):
        if self.precache:
            data = [self.dataset[j] for j in i]
        elif self.xy_tuple_form:
            data = self.dataset[i]
        if self.xy_tuple_form:
            x = [datum[0] for datum in data]
            if self.labels is not None:
                y = self.labels[i]
            else:
                y = [datum[1] for datum in data]
        else:
            x = self.dataset[0][i]
            y = self.dataset[1][i]


        return x, y

    def __getitem__(self, index, accessed_by_background=False):
        'Generate one batch of data'
        if not accessed_by_background and self.use_background_process:
            while self.queue.empty():
                time.sleep(0.1)
            return self.queue.get()

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        '''Overriding this to deal with None dimensions (i.e. variable length times)
        and allow for MultiProcessingDataset to work (only works for slices) '''

        x, y = self.get_x_y(list_IDs_temp)
        x = three_dim_pad(x, self.mask_value, max_length=self.max_length)
        if not self.time_first: # we want batch by feature by time
            x = x.transpose((0, 2,1, *[i + 3 for i in range(x.ndim - 3)]))
        if not hasattr(self, "class_type") or self.class_type == "nominal":
            y =  keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.class_type == "quantile":
            y = y
        return x, np.stack(y)


class RULEdfDataGenerator(EdfDataGenerator):
    '''
    Similar to EdfDataGenerator but runs random under_sampling each run
    '''
    def __init__(self, dataset, mask_value=-10000, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, class_type="nominal", shuffle=True, max_length=None, time_first=True, precache=False, xy_tuple_form=True, **kwargs):
        super().__init__(dataset=dataset, mask_value=mask_value, labels=labels, batch_size=batch_size, dim=dim, n_channels=n_channels,
                     n_classes=n_classes, class_type=class_type, shuffle=shuffle, max_length=max_length, time_first=time_first, precache=precache, xy_tuple_form=xy_tuple_form, **kwargs)
        self.full_indexes = self.list_IDs
        self.on_epoch_end()
    def on_epoch_end(self):

        'Updates indexes after each epoch and balances such that all data is used per epoch'
        if self.labels is not None:
            copiedSelfSampleInfo = Dict()
            oldIndicesByLabels = Dict()
            allLabels = Dict()
            for i in range(len(self.labels)):
                label = self.labels[i]
                if label not in oldIndicesByLabels.keys():
                    oldIndicesByLabels[label] = []
                    allLabels[label] = 0
                oldIndicesByLabels[label].append(i)
                allLabels[label] += 1

            min_label_count = min([allLabels[label] for label in allLabels.keys()])
            newInd = 0
            self.list_IDs = []
            for label in oldIndicesByLabels.keys():
                oldIndicesByLabels[label] = choice(oldIndicesByLabels[label], size=min_label_count, replace=False)
                for oldInd in oldIndicesByLabels[label]:
                    self.list_IDs.append(oldInd)


        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class DataGenMultipleLabels(EdfDataGenerator):
    '''
    To be able to deal with cases where neural net can predict for multiple things at once
    '''
    def __init__(self, dataset, mask_value=-10000, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=(2, 2), class_type="nominal", shuffle=True, max_length=None, time_first=True, precache=False, xy_tuple_form=True, num_labels=2, shuffle_channels=False, **kwargs):
        super().__init__( dataset, mask_value, labels, batch_size, dim, n_channels,
                     n_classes, class_type, shuffle, max_length, time_first, precache=precache, xy_tuple_form=xy_tuple_form, **kwargs)

        assert num_labels >= 2
        assert num_labels == len(n_classes)
        assert xy_tuple_form or len(labels) == num_labels
        self.num_labels = num_labels
        self.shuffle_channels = shuffle_channels


        if self.precache:
            self.dataset = dataset[:]

    def __getitem__(self, index, accessed_by_background=False):
        'Generate one batch of data'
        if not accessed_by_background and self.use_background_process:
            while self.queue.empty():
                time.sleep(0.1)
            return self.queue.get()
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def create_validation_train_split(self, validation_size=0.1, use_stratify=True):
        if not self.precache:
            raise Exception("NOT IMPLEMENTED")
        if use_stratify:
            train_data, validation_data, train_labels, validation_labels = train_test_split(self.dataset, self.labels, test_size=validation_size, stratify=self.labels)
        else:
            train_data, validation_data, train_labels, validation_labels = train_test_split(self.dataset, self.labels, test_size=validation_size)

        train_data_gen = DataGenMultipleLabels(num_labels=num_labels, dataset=train_data, mask_value=self.mask_value, labels=train_labels, batch_size=self.batch_size, dim=self.dim, n_channels=self.n_channels,
                     n_classes=self.n_classes, shuffle=self.shuffle, max_length=self.max_length, time_first=self.time_first, precache=self.precache, class_type=self.class_type, shuffle_channels=self.shuffle_channels)


        validation_data_gen = DataGenMultipleLabels(num_labels=num_labels, dataset=validation_data, mask_value=self.mask_value, labels=validation_labels, batch_size=self.batch_size, dim=self.dim, n_channels=self.n_channels,
                     n_classes=self.n_classes, shuffle=self.shuffle, max_length=self.max_length, time_first=self.time_first, precache=self.precache, class_type=self.class_type,  shuffle_channels=self.shuffle_channels)

        return train_data_gen, validation_data_gen


    def get_x_y(self, i):
        if self.precache:
            data = [self.dataset[j] for j in i]
        else:
            print("Using mp with dataset of size {}".format(len(i)))
            data = self.dataset[i]

        if self.xy_tuple_form:
            x = [datum[0] for datum in data]
            instanceLabels = [datum[1] for datum in data]
            labels = []
            for class_i in range(self.num_labels):
                labels.append([instanceLabel[class_i] for instanceLabel in instanceLabels])
        else:
            if len(data[0]) == 1:
                x = data
            else:
                x = [datum[0] for datum in data]
            labels = []
            for class_i in range(self.num_labels):
                labels.append([self.labels[class_i][j] for j in i])
        return x, labels

    def __data_generation(self, list_IDs_temp):
        x, labels = self.get_x_y(list_IDs_temp)
        x = three_dim_pad(x, self.mask_value, max_length=self.max_length)
        if self.shuffle_channels:
            new_col_order = [i for i in range(x.shape[2])]
            np.random.shuffle(new_col_order)
            x_new = x[:,:,new_col_order,:] #shuffles features, to do an experiment to see if column order is being memorized
        if not self.time_first: # we want batch by feature by time
            x = x.transpose((0, 2,1, *[i + 3 for i in range(x.ndim - 3)]))

        y_labels = []
        for i, sing_label in enumerate(labels):
            y =  keras.utils.to_categorical(sing_label, num_classes=self.n_classes[i])
            y_labels.append(y)

        return x, y_labels

class RULDataGenMultipleLabels(DataGenMultipleLabels):
    def __init__(self, dataset, mask_value=-10000, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=(2, 2), class_type="nominal", shuffle=True, max_length=None, time_first=True, precache=False, xy_tuple_form=True, num_labels=2, shuffle_channels=False, **kwargs):
        super().__init__(dataset=dataset, mask_value=mask_value, labels=labels, batch_size=batch_size, dim=dim, n_channels=n_channels,
                     n_classes=n_classes, class_type=class_type, shuffle=shuffle, max_length=max_length, time_first=time_first, precache=precache, xy_tuple_form=xy_tuple_form, num_labels=num_labels, shuffle_channels=shuffle_channels, **kwargs)
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch and balances such that all data is used per epoch'
        if self.labels is not None:
            copiedSelfSampleInfo = Dict()
            oldIndicesByLabels = Dict()
            allLabels = Dict()
            for i in range(len(self.labels[0])):
                label = self.labels[0][i] #use the first label
                if label not in oldIndicesByLabels.keys():
                    oldIndicesByLabels[label] = []
                    allLabels[label] = 0
                oldIndicesByLabels[label].append(i)
                allLabels[label] += 1

            min_label_count = min([allLabels[label] for label in allLabels.keys()])
            newInd = 0
            self.list_IDs = []
            for label in oldIndicesByLabels.keys():
                oldIndicesByLabels[label] = choice(oldIndicesByLabels[label], size=min_label_count, replace=False)
                for oldInd in oldIndicesByLabels[label]:
                    self.list_IDs.append(oldInd)


        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
