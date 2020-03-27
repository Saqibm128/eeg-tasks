import sys, os
sys.path.append(os.path.realpath(".."))
import sys, os
sys.path.append(os.path.realpath(".."))
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import random
from os import path
import tensorflow as tf
import data_reader as read
import util_funcs
import string

from addict import Dict
import sacred
from predictSeizureLSTM import ex as old_ex
ex = sacred.Experiment(name="seizure_long_term_autoencoder")
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))
import preprocessingV2.preprocessingV2 as ppv2
from keras_models.metrics import f1, sensitivity, specificity, auroc
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from keras_models.metrics import f1, sensitivity, specificity

@ex.config
def config():
    num_epochs=1000
    patience=20
    model_name = "/n/scratch2/ms994/out/" + util_funcs.randomString() + ".h5"
    lr = 0.0001
    lr_decay = 0.9
    steps_per_epoch =700 #approximate num epochs in balanced train set
    valid_steps_per_epoch = 200
    lstm_h = 128
    filter_1 = 8
    filter_2 = 4
    test_steps_per_epoch = 500
    verbose = 2

@ex.capture
def get_model(lr, lstm_h):
    inputX = tf.keras.layers.Input((9,21,1000,1))
    x = inputX
    # inputX = tf.keras.Input((21*1000*1,))
    # x = tf.keras.GaussianNoise(4)(inputX)
    # x = tf.keras.layers.Reshape((9,21,1000,1))(inputX)
    #encoder
    for i in range(6):
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        if i > 3:
            num_filters=filter_2
        else:
            num_filters=filter_1
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_filters, (3,3), activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((1,2)))(x)
    decoder_reshape_shape =  tf.keras.backend.int_shape(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    decoder_lstm_out_shape = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.CuDNNLSTM(lstm_h, return_sequences=True)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.CuDNNLSTM(int(lstm_h/2), return_sequences=True)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(x)
    encoder_out = x
    #seizure detection
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(x)
    y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation="relu"), name="seizure_detection")(x)

    #decoder
    x = tf.keras.layers.CuDNNLSTM(decoder_lstm_out_shape[2], return_sequences=True)(encoder_out)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Reshape(decoder_reshape_shape[1:])(x)
    for i in range(6):
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        if i > 3:
            num_filters=filter_1
        else:
            num_filters=filter_2
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(num_filters, (3,3), activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((1,2)))(x)

    y_decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(1, (1,1), activation="relu"), name="upsample")(x)
    # y_decoder.name = x
    # y_decoder = tf.keras.layers.Reshape((9,21,1000,1), name="decoder")(x)

    model = tf.keras.Model(inputs=[inputX], outputs=[y, y_decoder])
    model.compile(tf.keras.optimizers.Adam(lr=lr), loss={"seizure_detection":"categorical_crossentropy","upsample":"mse"}, metrics={"seizure_detection":["binary_accuracy", f1, sensitivity, specificity],"upsample":[]}, )
    return model

@ex.capture
def get_cb(lr, lr_decay, patience, model_name):
    return [tf.keras.callbacks.LearningRateScheduler(lambda x, lr: lr*lr_decay, verbose=1), tf.keras.callbacks.EarlyStopping(monitor="val_seizure_detection_f1", mode="max", patience=patience, verbose=True), tf.keras.callbacks.ModelCheckpoint(model_name, monitor="val_seizure_detection_f1", mode="max", save_best_only=True, verbose=True)]
    return cb

def read_tfrecord(example):
    features = {'original_index': tf.io.FixedLenFeature([1], tf.int64, ),\
               'data':  tf.FixedLenFeature([9*21*1000], tf.float32,),\
               'label':  tf.FixedLenFeature([10], tf.int64, [0 for i in range(10)]),\
               'subtypeLabel':  tf.FixedLenFeature([10], tf.int64, [0 for i in range(10)]),\
               'patient':  tf.FixedLenFeature([1], tf.int64,), \
               'session':  tf.FixedLenFeature([1], tf.int64,),
                       }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
#     return example
    data = tf.reshape(example['data'], [9,21,1000,1])
#     data = (example['data'])
    class_label = tf.cast(example['label'], tf.int32)

    paddings = tf.constant([[0,0],[0,0,], [42, 42], [0,0]])
    # 'constant_values' is 0.
    # rank of 't' is 2.
    data_decoder = tf.pad(data, paddings, "CONSTANT")

    return data, (tf.one_hot(class_label, 2)[0:9], data_decoder)

def get_batched_dataset(filenames):
    BATCH_SIZE=32

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=4)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=4)

#     dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(256)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(20) #

    return dataset
# featureData = read_tfrecord(allData[1])
import tensorflow as tf
def get_training_dataset():
    training_filenames = ["/n/scratch2/ms994/medium_size/train_1.tfrecords","/n/scratch2/ms994/medium_size/train_2.tfrecords","/n/scratch2/ms994/medium_size/train_3.tfrecords","/n/scratch2/ms994/medium_size/train_0.tfrecords"]
    return get_batched_dataset(training_filenames)

def get_validation_dataset():
    validation_filenames=["/n/scratch2/ms994/medium_size/valid_1.tfrecords","/n/scratch2/ms994/medium_size/valid_2.tfrecords","/n/scratch2/ms994/medium_size/valid_3.tfrecords","/n/scratch2/ms994/medium_size/valid_0.tfrecords"]
    return get_batched_dataset(validation_filenames)

def get_test_dataset():
    test_filenames=["/n/scratch2/ms994/medium_size/test_1.tfrecords","/n/scratch2/ms994/medium_size/test_2.tfrecords","/n/scratch2/ms994/medium_size/test_3.tfrecords","/n/scratch2/ms994/medium_size/test_0.tfrecords"]
    return get_batched_dataset(test_filenames)
negative_train_filenames =  ["/n/scratch2/ms994/medium_size/train_neg_1.tfrecords","/n/scratch2/ms994/medium_size/train_neg_2.tfrecords","/n/scratch2/ms994/medium_size/train_neg_3.tfrecords","/n/scratch2/ms994/medium_size/train_neg_0.tfrecords"]
positive_train_filenames =  ["/n/scratch2/ms994/medium_size/train_pos_1.tfrecords","/n/scratch2/ms994/medium_size/train_pos_2.tfrecords","/n/scratch2/ms994/medium_size/train_pos_3.tfrecords","/n/scratch2/ms994/medium_size/train_pos_0.tfrecords"]

def get_positive_train_dataset():
    return get_batched_dataset(positive_train_filenames)

def get_negative_train_dataset():
    return get_batched_dataset(negative_train_filenames)

def get_balanced_train_dataset(batch_size=64, max_queue_size=10):
    pos_ds = get_positive_train_dataset()
    neg_ds = get_negative_train_dataset()
    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

    return resampled_ds



@ex.main
def main(num_epochs, steps_per_epoch, valid_steps_per_epoch, test_steps_per_epoch, verbose, model_name):
    train = get_balanced_train_dataset()
    valid = get_validation_dataset()
    test = get_test_dataset()
    model = get_model()
    model.summary()
    # history = model
    history = model.fit(train, steps_per_epoch=steps_per_epoch, validation_data=valid,  validation_steps=valid_steps_per_epoch, epochs=num_epochs, callbacks=get_cb(),verbose=verbose)
    ex.add_artifact(model_name)
    best_model = tf.keras.models.load_model(model_name, custom_objects={"f1":f1,"sensitivity":sensitivity,"specificity":specificity}, compile=True)
    return (history.history, best_model.evaluate(test, steps=test_steps_per_epoch))

if __name__ == "__main__":
    ex.run_commandline()
