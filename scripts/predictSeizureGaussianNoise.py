import sys, os
sys.path.append(os.path.realpath("."))
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
tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()
import data_reader as read
import util_funcs
import string

from addict import Dict
import sacred
import preprocessingV2.preprocessingV2 as ppv2
from keras_models.metrics import f1, sensitivity, specificity, auc
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from addict import Dict
ex = sacred.Experiment(name="seizure_gaussian")

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

def read_tfrecord(example):
    features = { \
               'data':  tf.io.FixedLenFeature([21*1000], tf.float32,),\
               'label':  tf.io.FixedLenFeature([1], tf.int64,),\
               'subtypeLabel':  tf.io.FixedLenFeature([1], tf.int64,),\
               'session':  tf.io.FixedLenFeature([1], tf.int64,), \
               'montage':  tf.io.FixedLenFeature([22], tf.int64,)}

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
#     return example

    data = tf.reshape(example['data'], [1000,21,1])
    # data = (example['data'])

    class_label = tf.cast(example['label'], tf.int32)

    # del example
    return data, tf.one_hot(class_label[0], 2)

def read_random_tfrecord(example):
    features = { \
               'data':  tf.io.FixedLenFeature([21*1000], tf.float32,),\
               'label':  tf.io.FixedLenFeature([1], tf.int64,),\
               'subtypeLabel':  tf.io.FixedLenFeature([1], tf.int64,),\
               'session':  tf.io.FixedLenFeature([1], tf.int64,), \
               'montage':  tf.io.FixedLenFeature([22], tf.int64,)}

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
#     return example

    data = tf.reshape(example['data'], [1000,21,1])
    # data = (example['data'])

    class_label = tf.cast(example['label'], tf.int32)

    # del example
    return data, tf.one_hot(class_label[0], 2)

@ex.capture
def get_batched_dataset(filenames, batch_size, random_rearrange_each_batch, max_queue_size=10,  n_process=4, is_train=False):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
#
    if is_train and random_rearrange_each_batch:
        dataset = dataset.map(read_random_tfrecord, num_parallel_calls=n_process)
    else:
        dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)
#     dataset = dataset.cache() # IF this dataset fits in RAM
    dataset = dataset.repeat()
#     if is_train:
#         resampler = tf.data.experimental.rejection_resample(lambda x, y: y, target_dist=[1, 1])
#         dataset = dataset.apply(resampler)
    if is_train:
        dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_train:
        dataset = dataset.prefetch(max_queue_size)
    else:
        dataset = dataset.prefetch(int(max_queue_size/4)) #store a lot less for the other sets to avoid wasting memory

    return dataset

@ex.capture
def get_positive_train_dataset(filenames, random_rearrange_each_batch, max_queue_size=5, n_process=4, is_train=True):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    if is_train and random_rearrange_each_batch:
        dataset = dataset.map(read_random_tfrecord, num_parallel_calls=n_process)
    else:
        dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)

    dataset = dataset.filter(lambda x, y: tf.equal(tf.argmax(tf.cast(y, tf.int32), 0), 1))


#     dataset = dataset.cache() # IF this dataset fits in RAM
    dataset = dataset.repeat()
#     if is_train:
#         resampler = tf.data.experimental.rejection_resample(lambda x, y: y, target_dist=[1, 1])
#         dataset = dataset.apply(resampler)
    if is_train:
        dataset = dataset.shuffle(256)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_train:
        dataset = dataset.prefetch(max_queue_size)
    else:
        dataset = dataset.prefetch(int(max_queue_size/4)) #store a lot less for the other sets to avoid wasting memory

    return dataset

@ex.capture
def get_negative_train_dataset(filenames, random_rearrange_each_batch, max_queue_size=10, n_process=4, is_train=True):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    if is_train and random_rearrange_each_batch:
        dataset = dataset.map(read_random_tfrecord, num_parallel_calls=n_process)
    else:
        dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)


    dataset = dataset.filter(lambda x, y: tf.equal(tf.argmax(tf.cast(y, tf.int32), 0), 0))


#     dataset = dataset.cache() # IF this dataset fits in RAM
    dataset = dataset.repeat()
#     if is_train:
#         resampler = tf.data.experimental.rejection_resample(lambda x, y: y, target_dist=[1, 1])
#         dataset = dataset.apply(resampler)
    if is_train:
        dataset = dataset.shuffle(256)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_train:
        dataset = dataset.prefetch(max_queue_size)
    else:
        dataset = dataset.prefetch(int(max_queue_size/4)) #store a lot less for the other sets to avoid wasting memory
    return dataset

@ex.capture
def get_balanced_dataset(filenames, batch_size, max_queue_size=10, n_process=4, is_train=True):
    pos_ds = get_positive_train_dataset(filenames)
    neg_ds = get_negative_train_dataset(filenames)
    dataset = tf.data.experimental.sample_from_datasets([neg_ds, pos_ds], [0.5, 0.5])
    if is_train:
        dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(max_queue_size)

@ex.capture
def get_count_test_train(train_tfr, valid_tfr, test_tfr):
    num_train = 0
    for record in tf.python_io.tf_record_iterator(train_tfr):
         num_train += 1
    num_valid = 0
    for record in tf.python_io.tf_record_iterator(valid_tfr):
         num_valid += 1
    num_test = 0
    for record in tf.python_io.tf_record_iterator(test_tfr):
         num_test += 1
    return num_train, num_valid, num_test

@ex.capture
def get_model(g_noise, num_cnn_layers, num_lstm_layers, num_lin_layers, lstm_h, cnn2d_n_k, lin_h, lr):

    inputLayer = tf.keras.layers.Input((1000,21, 1))
    x = inputLayer
    x = tf.keras.layers.GaussianNoise(g_noise)(x)
    for i in range(num_cnn_layers):
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.Conv2D(cnn2d_n_k, (3,3))(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPool2D((2,1))(x)
    x = tf.keras.layers.Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3])))(x)
    for j in range(num_lstm_layers):
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.CuDNNLSTM(lstm_h, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    for k in range(num_lin_layers):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(lin_h)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, name="seizure", activation="softmax")(x)
    model = tf.keras.Model(inputs=inputLayer, outputs=x)
    model.compile(tf.keras.optimizers.Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy", f1, auc, sensitivity, specificity])
    model.summary()
    return model
@ex.config
def config():
    train_tfr = "/n/scratch2/ms994/train_4s.tfr"
    valid_tfr = "/n/scratch2/ms994/valid_4s.tfr"
    test_tfr = "/n/scratch2/ms994/test_4s.tfr"
    n_process=3
    batch_size=128
    g_noise=1
    num_cnn_layers=3
    num_lstm_layers=1
    num_lin_layers=3
    lstm_h = 32
    cnn2d_n_k = 4
    lin_h = 100
    lr = 0.00005
    patience=20
    train_steps=int(203894/batch_size)
    valid_steps=int(81251/batch_size)
    test_steps=int(129670/batch_size)
    seizure_class_weight=12 #seizure is 1/12 of train dataset, balance out with this
    model_name = util_funcs.randomString() +".h5"
    early_stopping_on = "val_loss"
    mode="auto"
    num_epochs=100
    random_rearrange_each_batch=False
    verbose=2
    reduce_lr_patience=5

@ex.capture
def get_model_checkpoint(model_name, early_stopping_on, mode):
    return tf.keras.callbacks.ModelCheckpoint(model_name, monitor=early_stopping_on, save_best_only=True, verbose=1, mode=mode)
@ex.capture
def get_early_stopping(patience, early_stopping_on, mode):
    return tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1, min_delta=0.001, monitor=early_stopping_on, mode=mode)
@ex.capture
def get_reduce_lr(early_stopping_on, reduce_lr_patience, mode):
    return tf.keras.callbacks.ReduceLROnPlateau(monitor=early_stopping_on, patience=reduce_lr_patience, mode=mode)
@ex.capture
def get_cb_list():
    return [get_model_checkpoint(), get_early_stopping(), get_reduce_lr()]

@ex.main
def main(train_steps, valid_steps, test_steps, model_name, num_epochs, verbose):
    train_unbalanced = get_batched_dataset(["/n/scratch2/ms994/train_4s.tfr"],  is_train=True)
    valid_data = get_batched_dataset(["/n/scratch2/ms994/valid_4s.tfr"], is_train=False)
    test_data = get_batched_dataset(["/n/scratch2/ms994/test_4s.tfr"],  is_train=False)
    model = get_model()
    history = model.fit(
        train_unbalanced,
        steps_per_epoch=(train_steps),
        validation_data=valid_data,
        validation_steps=(valid_steps),
        epochs=num_epochs,
        callbacks=get_cb_list(),
        verbose=verbose)
    model = tf.keras.models.load_model(model_name, custom_objects={"f1":f1, "auc":auc, "sensitivity":sensitivity, "specificity":specificity})
    loss, accuracy, f1_test, auc_test, sensitivity_test, specificity_test = model.evaluate(test_data, steps=test_steps)
    return {
        "history": history.history,
        "loss": loss,
        "acc": accuracy,
        "f1": f1_test,
        "auc": auc_test,
        "sens": sensitivity_test,
        "spec": specificity_test
    }

if __name__ == "__main__":
    ex.run_commandline()
