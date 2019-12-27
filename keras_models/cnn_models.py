#https://www.nature.com/articles/s41598-018-21495-7
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Dropout, MaxPool2D, Conv3D, Flatten, LeakyReLU, BatchNormalization, Concatenate

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

def inception_like_layer(x, num_filters):
    y0 = Conv2D(1, (1,1), activation="relu", padding='same')(x)
    y0 = Conv2D(num_filters, (2,1), activation="relu", padding='same')(y0)
    y1 = Conv2D(1, (1,1), activation="relu", padding='same')(x)
    y1 = Conv2D(num_filters, (3,1), activation="relu", padding='same')(y1)
    y2 = Conv2D(1, (1,1), activation="relu", padding='same')(x)
    y2 = Conv2D(num_filters, (5,1), activation="relu", padding='same')(y2)
    y3 = MaxPool2D(pool_size=(3, 1), strides=(1,1), padding='same')(x)
    y3 = Conv2D(num_filters, (1,1), activation="relu", padding='same')(y3)
    return Concatenate()([y0, y1, y2, y3])





def inception_like_pre_layers(input_shape=None, x=None, num_layers=4, max_pool_size=(1,2), max_pool_stride=(1,2), dropout=0.5, num_filters=30, get_kernel_regularizer=None, get_activity_regularizer=None):
    def get_none():
        return None
    if get_kernel_regularizer is None:
        get_kernel_regularizer = get_none
    if get_activity_regularizer is None:
        get_activity_regularizer = get_none
    if x is None:
        x = Input(input_shape)
    y = x
    for i in range(num_layers):
        y = BatchNormalization()(y)
        y = inception_like_layer(y, num_filters)
        y = MaxPool2D(max_pool_size, strides=max_pool_stride)(y)
        if dropout != 0:
            y = Dropout(dropout)(y)
    y = Conv2D(max(int(num_filters/4), 1), (1,1))(y)

    return x, y


    y0 = Conv2D(num_filters, (2,2),  activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(x)
    y0 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    max_additional_layers = num_layers - 1
    for i in range(max_additional_layers):
        y0 = Conv2D(int(num_filters/2) + 1, (1,1), activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y0)
        y0 = Conv2D(num_filters * 2, (2,2), activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y0)
        if i > max_additional_layers - 5: #add up to 5 max pools, to avoid negative dim
            y0 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y0)
        y0 = Dropout(dropout)(y0)
        y0 = BatchNormalization()(y0)
    y0 = Flatten()(y0)

    y1 = Conv2D(num_filters, (3,3),  activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(x)
    y1 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y1)
    y1 = Dropout(dropout)(y1)
    y1 = BatchNormalization()(y1)
    for i in range(max_additional_layers):
        y1 = Conv2D(int(num_filters/2) + 1, (1,1), activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y1)
        y1 = Conv2D(num_filters * 2, (3,3), activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y1)
        if i > max_additional_layers - 5: #add up to 5 max pools, to avoid negative dim
            y1 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y1)
        y1 = Dropout(dropout)(y1)
        y1 = BatchNormalization()(y1)
    y1 = Conv2D(int(num_filters/4) + 1, (1,1), activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y1)
    y1 = Flatten()(y1)

    y2 = Conv2D(num_filters, (4,4),  activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(x)
    y2 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y2)
    y2 = Dropout(dropout)(y2)
    y2 = BatchNormalization()(y2)
    for i in range(max_additional_layers):
        y2 = Conv2D(int(num_filters / 2) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y2)
        y2 = Conv2D(num_filters * 2, (4,4), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y2)
        if i > max_additional_layers - 5: #add up to 5 max pools, to avoid negative dim
            y2 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y2)
        y2 = Dropout(dropout)(y2)
        y2 = BatchNormalization()(y2)
    y2 = Conv2D(int(num_filters / 4) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y2)
    y2 = Flatten()(y2)

    y3 = Conv2D(num_filters, (5,5),  activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(x)
    y3 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y3)
    y3 = Dropout(dropout)(y3)
    y3 = BatchNormalization()(y3)
    for i in range(max_additional_layers):
        y3 = Conv2D(int(num_filters/2) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y3)

        y3 = Conv2D(num_filters * 2, (5,5), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y3)
        if i > max_additional_layers - 5: #add up to 5 max pools, to avoid negative dim
            y3 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y3)
        y3 = Dropout(dropout)(y3)
        y3 = BatchNormalization()(y3)
    y3 = Conv2D(int(num_filters/4) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y3)

    y3 = Flatten()(y3)

    y4 = Conv2D(num_filters, (6,6),  activation="relu", kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(x)
    y4 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y4)
    y4 = Dropout(dropout)(y4)
    y4 = BatchNormalization()(y4)
    for i in range(max_additional_layers):
        y4 = Conv2D(int(num_filters/2) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y4)
        y4 = Conv2D(num_filters * 2, (6,6), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y4)
        if i > max_additional_layers - 5: #add up to 5 max pools, to avoid negative dim
            y4 = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(y4)
        y4 = Dropout(dropout)(y4)
        y4 = BatchNormalization()(y4)
    y4 = Conv2D(int(num_filters/4) + 1, (1,1), activation="relu", padding='same', kernel_regularizer=get_kernel_regularizer(), activity_regularizer=get_activity_regularizer())(y4)
    y4 = Flatten()(y4)
    y = Concatenate()([y0, y1, y2, y3, y4])
    return x, y


def inception_like(input_shape, num_layers=4, max_pool_size=(1,2), max_pool_stride=(1,2), dropout=0.5, num_filters=30, output_activation='softmax',num_outputs=2):
    x, y = inception_like_pre_layers(input_shape, num_layers=num_layers, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride, dropout=dropout, num_filters=num_filters,)
    y = Dense(units=num_outputs, activation=output_activation)(y)
    model = Model(inputs=x, outputs =y)
    return model

def conv2d_gridsearch_pre_layers(
    dropout=0.25,
    x = None,
    input_shape=(None),
    num_conv_spatial_layers=1,
    num_conv_temporal_layers=1,
    conv_spatial_filter=(3,3),
    num_spatial_filter=10,
    conv_temporal_filter=(1,3),
    num_temporal_filter=10,
    max_pool_size=(2,2),
    max_pool_stride=(2,2),
    time_convolutions_first=False,
    max_pool_size_time=None,
    max_pool_stride_time=None,
    use_batch_normalization=False):

    if x is None:
        input = Input(shape=input_shape)
        x = input
    else:
        input = x
    if max_pool_size_time is None:
        max_pool_size_time = max_pool_size

    def get_time_layers(x):
        for i in range(num_conv_temporal_layers):
            if use_batch_normalization:
                x = BatchNormalization()(x)
            x = Conv2D(num_temporal_filter, conv_temporal_filter, padding='same', activation='relu')(x)
            x = MaxPool2D(pool_size=max_pool_size_time, strides=max_pool_stride_time)(x)
            x = Dropout(dropout)(x)
        return x

    if time_convolutions_first:
        x = get_time_layers(x)
    #should act to primarily convolve space, with some convolving of time
    for i in range(num_conv_spatial_layers):
        x = Conv2D(num_spatial_filter, conv_spatial_filter, input_shape=input_shape, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=max_pool_size, strides=max_pool_stride)(x) #don't break! 2^4 = 16
        if dropout != 0:
            x = Dropout(dropout)(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
    if not time_convolutions_first:
        x = get_time_layers(x)

    # x = Flatten()(x)
    return input, x

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
    use_batch_normalization=False,
    output_activation="softmax",
    num_outputs=2
    ):

    input, x = conv2d_gridsearch_pre_layers(
        dropout=dropout,
        input_shape=input_shape,
        num_conv_spatial_layers=num_conv_spatial_layers,
        num_conv_temporal_layers=num_conv_temporal_layers,
        conv_spatial_filter=conv_spatial_filter,
        num_spatial_filter=num_spatial_filter,
        conv_temporal_filter=conv_temporal_filter,
        num_temporal_filter=num_temporal_filter,
        max_pool_size=max_pool_size,
        max_pool_stride=max_pool_stride,
        use_batch_normalization=use_batch_normalization)

    x = Dense(activation=output_activation, units=num_outputs)(Flatten(x))
    return Model(input=input, outputs=x)

def vp_conv2d(dropout=0.25, input_shape=(None), filter_size=100, use_batch_normalization=False, max_pool_size=(1,2), output_activation="", num_outputs=2):
    inputs = Input(shape=input_shape)
    x = Conv2D(filter_size, (3,3), input_shape=input_shape, name="conv1", activation="relu", padding="same")(inputs)
    x = MaxPool2D(pool_size=max_pool_size, name="maxpool1")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size, (3,3), name="conv2", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=max_pool_size, name="maxpool2")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (2,3), name="conv3", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=max_pool_size, name="maxpool3")(x)
    x = Dropout(dropout)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,7), name="conv4", activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=max_pool_size, name="maxpool4")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,3), name="conv5", activation="relu", padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(filter_size*3, (1,3), name="conv6", activation="relu", padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(activation=output_activation, units=num_outputs)(x)

    return Model(input=inputs, outputs=x)
