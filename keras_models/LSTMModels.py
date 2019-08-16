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
