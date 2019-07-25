#https://www.nature.com/articles/s41598-018-21495-7
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Dropout, MaxPool2D, Conv3D, Flatten, LeakyReLU, BatchNormalization

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

def conv2d_gridsearch(
    dropout=0.25,
    input_shape=(None),
    num_conv_spatial_layers=1,
    num_conv_temporal_layers=1,
    conv_spatial_filter=(3,3),
    num_spatial_filter=100,
    conv_temporal_filter=(1,3),
    num_temporal_filter=300,
    max_pool_size=(2,2),
    max_pool_stride=(1,2),
    use_batch_normalization=False
    ):
    max_temporal_filter=max(num_temporal_filter, num_spatial_filter*3)
    input = Input(shape=input_shape)
    x = input

    #should act to primarily convolve space, with some convolving of time
    for i in range(num_conv_spatial_layers):
        if i == 2:
            num_spatial_filter *= 3 #match van_putten magic
        x = Conv2D(num_spatial_filter, conv_spatial_filter, input_shape=input_shape, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(x) #don't break! 2^4 = 16
        x = Dropout(dropout)(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
    for i in range(num_conv_temporal_layers):
        x = Conv2D(num_temporal_filter, conv_temporal_filter, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(x)
        x = Dropout(dropout)(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)

    # final conv layer to match VP arch, then flatten and run through dense output
    x = Conv2D(num_temporal_filter, (1,3), padding='same', activation='relu')(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(activation='softmax', units=2)(x)


    return Model(input=input, outputs=x)

def vp_conv2d(dropout=0.25, input_shape=(None), filter_size=100, use_batch_normalization=False):
    inputs = Input(shape=input_shape)
    x = Conv2D(filter_size, (3,3), input_shape=input_shape, name="conv1", activation="relu", padding="same")(inputs)
    x = MaxPool2D(pool_size=(2, 2), name="maxpool1")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size, (3,3), name="conv2", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=(2, 2), name="maxpool2")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (2,3), name="conv3", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=(2, 2), name="maxpool3")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,7), name="conv4", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=(1, 2), name="maxpool4")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,3), name="conv5", activation="relu", padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,3), name="conv6", activation="relu", padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(activation='softmax', units=2)(x)

    return Model(input=inputs, outputs=x)
