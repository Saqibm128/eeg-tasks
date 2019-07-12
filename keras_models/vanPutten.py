#https://www.nature.com/articles/s41598-018-21495-7
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Dropout, MaxPool2D

def vp_conv2d(dropout=0.25):
    layers = [
        Conv2D(100, (3,3)),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(dropout),

        Conv2D(100, (3,3)),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(dropout),

        Conv2D(300, (2,3)),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(dropout),

        Conv2D(300, (1,7)),
        Activation('relu'),
        MaxPool2D(pool_size=(1, 2)),
        Dropout(dropout),

        Conv2D(300, (1,3)),
        Conv2D(300, (1,3)),
        Dense(activation='softmax', output=2)
    ]
    return Sequential(layers)
