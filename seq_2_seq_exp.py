from sacred import Experiment
ex = Experiment(name="seq_2_seq_clustering")
import sys, os
import util_funcs
from copy import deepcopy as cp
import data_reader as read
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense




@ex.config
def config():
    num_files = 2
    n_process = 8
    latent_dim = 100
    freq_bins = read.EdfFFTDatasetTransformer.freq_bins
    input_shape = 21 * len(freq_bins) #num channels times number of freq bins we extrapolated out
    window_size = 10 #seconds
    non_overlapping = True
    num_epochs = 10
    batch_size = 10
    precache = True
    validation_split = 0.2

@ex.capture
def get_data(n_process, num_files, window_size, non_overlapping, precache):
    edfRawData = read.EdfDataset("train", "01_tcp_ar", num_files=num_files, n_process=n_process)
    edfFFTData = read.EdfFFTDatasetTransformer(edfRawData, window_size=pd.Timedelta(seconds=window_size), non_overlapping=non_overlapping, precache=precache, n_process=n_process)
    seq2seqData = read.Seq2SeqFFTDataset(edfFFTData, n_process=n_process)
    return np.asarray(seq2seqData[:])


@ex.capture
def create_model(input_shape, latent_dim):
    encoder_inputs = Input(shape=(None, input_shape))
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

@ex.main
def main():
    print("hi")
    data = get_data()
    model = create_model()
    model.fit([data, data], data)

if __name__ == "__main__":
    ex.run_commandline()
