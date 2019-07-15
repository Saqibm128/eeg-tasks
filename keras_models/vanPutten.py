#https://www.nature.com/articles/s41598-018-21495-7
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Dropout, MaxPool2D, Conv3D, Flatten, LeakyReLU

def simplified_vp_conv2d(dropout=0.25, input_shape=(None)):
    layers = [
        # Conv3D(100, (3,3,3), input_shape=input_shape, padding='same'),
        # Activation('sigmoid'),
        # Dropout(dropout),

        # Conv3D(100, (10,3,3), padding='same'),
        # Activation('sigmoid'),
        # Dropout(dropout),
        #
        # Conv3D(300, (10,3,3), padding='same'),
        # Activation('sigmoid'),
        # Dropout(dropout),
        #
        # Conv3D(300, (10,3,3), padding='same'),
        # Activation('sigmoid'),
        # Dropout(dropout),
        #
        # Conv3D(300, (10,3,3), padding='same'),
        # Activation('sigmoid'),
        # Dropout(dropout),


        #
        # Conv3D(300, (10,3,3), padding='same'),
        Flatten(input_shape=input_shape),
        Dense(activation='sigmoid', units=2)
    ]
    return Sequential(layers)

def vp_conv2d(dropout=0.25, input_shape=(None)):
    layers = [
        Conv2D(100, (3,3), input_shape=input_shape, padding='valid', name="conv1"),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), name="maxpool1"),
        Dropout(dropout),

        Conv2D(100, (3,3), name="conv2"),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), name="maxpool2"),
        Dropout(dropout),

        Conv2D(300, (2,3), name="conv3"),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), name="maxpool3"),
        Dropout(dropout),

        Conv2D(300, (1,7), name="conv4"),
        Activation('relu'),
        MaxPool2D(pool_size=(1, 2), name="maxpool4"),
        Dropout(dropout),

        Conv2D(300, (1,3), name="conv5"),
        Conv2D(300, (1,3), name="conv6"),
        Flatten(),
        Dense(activation='softmax', units=2)
    ]
    return Sequential(layers)
