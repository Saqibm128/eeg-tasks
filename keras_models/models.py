import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation

def get_simple_lstm(input_shape, latent_shape, ffn_nodes, num_lstm, num_feed_forward, activation="relu"):
    lstm = Input(shape=(None, input_shape)) #arbitrary time steps by num_features
    for i in range(num_lstm - 1):
        lstm = LSTM(latent_shape, return_state=False)(lstm)
        lstm = Activation(activation)(lstm)
    for i in range(num_feed_forward):
        lstm = Dense(ffn_nodes)(lstm)



def get_seq_2_seq(input_shape, latent_shape):
    encoder_inputs = Input(shape=(None, input_shape)) #arbitrary time steps by num features
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, input_shape))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(input_shape, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], [decoder_outputs, state_h, state_c])
    model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', None, None])
    return model

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ArbitraryDataGenerator(keras.utils.Sequence):
    """

    'Generates data for Keras'
    For EEG data samples, trying to pad_pack the entire sequence with the
    longest length data segment is memory inefficient. If there are 3k instances,
    the longest is about 100 times longer than the average, but most instances are
    smaller, than why make such a sparse tensor?

     Instead, let's generate a pad_packed sequence dynamically for each batch.
     Batches containing the longest sequences can be necessarily larger tensors,
     but otherwise don't allocate extreme amounts of memory

    Parameters
    ----------
    list_IDs : list
        id strings, usually filenames for this project
    labels : list
        if set to None, we just return X
    batch_size : int
        how big a batch size to return
    seq_like : Sequence
        Anything that implements _len_ and __getitem__
        Some of the datasets I assumed would be hard to fit in memory
    dim : tuple
        for time data, is (time_steps, features)
    n_channels : int
        Description of parameter `n_channels`.
    n_classes : int
        Description of parameter `n_classes`.
    shuffle : bool
        If True, we shuffle the instances in each batch

    Attributes
    ----------
    on_epoch_end : type
        Description of attribute `on_epoch_end`.
    dim
    batch_size
    labels
    list_IDs
    n_channels
    n_classes
    shuffle

    """
    def __init__(self, list_IDs, labels,
                 seq_like, batch_size=32, dim=(32,32,32),
                 n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.seq_like = seq_like
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

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
