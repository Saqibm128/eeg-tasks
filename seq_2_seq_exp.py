from sacred import Experiment
ex = Experiment(name="seq_2_seq_clustering")
import sys, os
import util_funcs
from copy import deepcopy as cp
import data_reader as read
import pandas as pd

@ex.config
def config():
    num_files = None
    n_process = 8
    latent_dim = 100
    input_shape = 30*99

@ex.capture
def get_data(n_process, num_files):
    edfRawData = read.EdfDataset("train", "01_tcp_ar", num_files=num_files, n_process=n_process)
    edfFFTData = read.EdfFFTDatasetTransformer(edfRawData, window_size=pd.Timedelta(seconds=1), precache=False, n_process=7)

    fftData = edfFFTData[0:1]


@ex.capture
def create_model(input_shape, latent_dim):
    # From https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    # Define an input sequence and process it.
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
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
