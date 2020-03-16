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
ex = sacred.Experiment(name="seizure_long_term")
# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))
import preprocessingV2.preprocessingV2 as ppv2
from keras_models.metrics import f1
from sklearn.metrics import f1_score, roc_auc_score, classification_report

@ex.capture
def read_tfrecord(example, multilabel):
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
    # data = (example['data'])

    class_label = tf.cast(example['label'], tf.int32)

    if multilabel:
        return data, (tf.one_hot(class_label[:9], 2), tf.one_hot(tf.cast(tf.reduce_any(tf.cast(class_label[:9], tf.bool), tf.int32), 2)))
    else:
        return data, tf.one_hot(class_label[:9], 2)

@ex.capture
def get_batched_dataset(filenames, batch_size, use_fft, max_queue_size, max_std, n_process, is_train=True):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)
    if is_train and max_std != None:
        dataset = dataset.filter(lambda x, y: tf.reduce_all(tf.math.reduce_std(x, axis=0) < max_std))
    if use_fft:
        dataset = dataset.map(lambda x, y: ( \
                                            tf.cast(tf.signal.fft( tf.cast(x, tf.complex64)), tf.float64), \
                                            y), num_parallel_calls=n_process)
#     dataset = dataset.cache() # IF this dataset fits in RAM
    dataset = dataset.repeat()
    if is_train:
        dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(max_queue_size)
    return dataset

@ex.capture
def get_valid_index(valid_pkl_20s_index):
    global valid_index
    if valid_index is not None:
        return valid_index
    data = pkl.load(open(valid_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    valid_index = data
    return data

train_index = None
valid_index = None
test_index = None

@ex.capture
def get_train_index(train_pkl_20s_index):
    global train_index
    if train_index is not None:
        return train_index
    data = pkl.load(open(train_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    train_index = data
    return data

@ex.capture
def get_valid_index(valid_pkl_20s_index):
    global valid_index
    if valid_index is not None:
        return valid_index
    data = pkl.load(open(valid_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    valid_index = data
    return data
@ex.capture
def get_test_index(test_pkl_20s_index):
    global test_index
    if test_index is not None:
        return test_index
    data = pkl.load(open(test_pkl_20s_index, "rb"))
    for i in data.keys():
        data[i].original_ind = i
    test_index = data
    return data

@ex.capture
def get_test_labels():
    test_index = get_test_index()
    #made a mistake, it should be the first 9 instead
    return np.array([test_index[key].time_seizure_label[:9] for key in test_index.keys()])
@ex.capture
def get_full_training_dataset(training_filenames):
    return get_batched_dataset(training_filenames)

@ex.capture
def get_validation_dataset(validation_filenames):
    return get_batched_dataset(validation_filenames, is_train=False)

@ex.capture
def get_test_dataset(test_filenames):
    return get_batched_dataset(test_filenames, is_train=False)

@ex.capture
def get_positive_train_dataset(positive_train_filenames):
    return get_batched_dataset(positive_train_filenames)

@ex.capture
def get_negative_train_dataset(negative_train_filenames):
    return get_batched_dataset(negative_train_filenames)

@ex.capture
def get_balanced_train_dataset(batch_size, max_queue_size):
    pos_ds = get_positive_train_dataset()
    neg_ds = get_negative_train_dataset()
    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

    return resampled_ds

@ex.capture
def get_train_dataset(train_dataset_mode):
    if train_dataset_mode == "full":
        return get_full_training_dataset()
    elif train_dataset_mode == "under":
        return get_balanced_train_dataset()


@ex.named_config
def undersample():
    total_train_len = 4526*2 #there are 4526 positives
    train_dataset_mode = "under"

@ex.named_config
def oversample():
    train_dataset_mode = "under"
    total_train_len=67551 #these are the total number of instances in train set


@ex.capture
def get_validation_steps_per_epoch(total_valid_len, batch_size):
    return int(np.ceil(total_valid_len/batch_size))

@ex.capture
def get_test_steps_per_epoch(total_test_len, batch_size):
    return int(np.ceil(total_test_len/batch_size))

@ex.capture
def get_steps_per_epoch(total_train_len, batch_size, steps_per_epoch):
    if steps_per_epoch is not None:
        return steps_per_epoch
    else:
        return int(np.ceil(total_train_len/batch_size))
# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@ex.named_config
def remove_outlier_by_std_thresh():
    max_std = 50.0/10**6 #mne transformed to 1*10^-6 V from µV
@ex.capture
def get_gaussian_noise(gaussian_noise):
    return gaussian_noise * 10**(-6)
@ex.config
def config():
    model_name = "/n/scratch2/ms994/out/" + randomString() + ".h5"
    negative_train_filenames =  ["/n/scratch2/ms994/medium_size/train_neg_1.tfrecords","/n/scratch2/ms994/medium_size/train_neg_2.tfrecords","/n/scratch2/ms994/medium_size/train_neg_3.tfrecords","/n/scratch2/ms994/medium_size/train_neg_0.tfrecords"]
    positive_train_filenames =  ["/n/scratch2/ms994/medium_size/train_pos_1.tfrecords","/n/scratch2/ms994/medium_size/train_pos_2.tfrecords","/n/scratch2/ms994/medium_size/train_pos_3.tfrecords","/n/scratch2/ms994/medium_size/train_pos_0.tfrecords"]
    training_filenames = ["/n/scratch2/ms994/medium_size/train_1.tfrecords","/n/scratch2/ms994/medium_size/train_2.tfrecords","/n/scratch2/ms994/medium_size/train_3.tfrecords","/n/scratch2/ms994/medium_size/train_0.tfrecords"]
    test_filenames=["/n/scratch2/ms994/medium_size/test_1.tfrecords","/n/scratch2/ms994/medium_size/test_2.tfrecords","/n/scratch2/ms994/medium_size/test_3.tfrecords","/n/scratch2/ms994/medium_size/test_0.tfrecords"]
    validation_filenames=["/n/scratch2/ms994/medium_size/valid_1.tfrecords","/n/scratch2/ms994/medium_size/valid_2.tfrecords","/n/scratch2/ms994/medium_size/valid_3.tfrecords","/n/scratch2/ms994/medium_size/valid_0.tfrecords"]
    batch_size=16
    train_pkl_20s_index="/n/scratch2/ms994/medium_size/train/20sIndex.pkl"
    valid_pkl_20s_index="/n/scratch2/ms994/medium_size/valid/20sIndex.pkl"
    test_pkl_20s_index="/n/scratch2/ms994/medium_size/test/20sIndex.pkl"
    total_train_len=67551 #these are the total number of instances in train set
    steps_per_epoch=None
    total_valid_len=15594
    total_test_len=24097
    randomly_reorder_channels = False
    filter_size=(3,3)
    train_dataset_mode = "full"
    max_pool_size = (1,2)
    max_std = None
    num_filters=16
    num_layers=6
    lstm_h=256
    post_lin_h =256
    num_lin_layers=2
    gaussian_noise=2
    dropout = 0.5
    n_process = 8
    num_epochs=1000
    max_queue_size = 30
    lr = 0.00001
    lr_decay = 0.5
    patience=20
    verbose=2
    use_fft = False
    use_bidirection = False
    multilabel = False

@ex.capture
def getCachedData():
    testDR = ppv2.FileDataReader(split="test", directory="/n/scratch2/ms994/medium_size/test", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/test/20sIndex.pkl", "rb")))
    trainDR = ppv2.RULDataReader(split="train", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/train/20sIndex.pkl", "rb")))
    validDR = ppv2.FileDataReader(split="valid", directory="/n/scratch2/ms994/medium_size/valid", cachedIndex=pkl.load(open("/n/scratch2/ms994/medium_size/valid/20sIndex.pkl", "rb")))
    return trainDR, validDR, testDR

@ex.capture
def get_model(num_filters, filter_size, use_bidirection, gaussian_noise, multilabel, num_layers, max_pool_size, lstm_h, num_lin_layers, post_lin_h, use_fft, dropout):
    input = tf.keras.layers.Input((9,21,1000,1), name="input")
    if not use_fft:
        x = tf.keras.layers.GaussianNoise(get_gaussian_noise())(input)
        for i in range(num_layers):
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_filters, filter_size, activation="relu"))(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(max_pool_size))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.BatchNormalization()(x)
    else: #dont do cnn on fft data
        x = input
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    lstm = tf.keras.layers.LSTM(lstm_h, activation="relu", return_sequences=True)
    if use_bidirection:
        x = tf.keras.layers.Bidirectional(lstm)(x)
    else:
        x = lstm(x)
    for i in range(num_lin_layers):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(post_lin_h, activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    y_time = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation="relu"), name="over_time")(x)
    x = tf.keras.layers.Flatten()(x)
    y_overall = tf.keras.layers.Dense(2, activation="relu")(x)
    if multilabel:
        model = tf.keras.Model(inputs=[input], outputs=[y_time, y_overall]) #todo figure out the loss_weights
        model.compile(tf.keras.optimizers.Adam(lr=0.0001), loss=["categorical_crossentropy", "categorical_crossentropy"], loss_weights=[1,1], metrics=["binary_accuracy", f1])
    else:
        model = tf.keras.Model(inputs=[input], outputs=[y_time]) #todo figure out the loss_weights
        model.compile(tf.keras.optimizers.Adam(lr=0.0001), loss=["categorical_crossentropy"],  metrics=["binary_accuracy", f1])
    return model

@ex.capture
def get_callbacks(lr, lr_decay, patience, model_name):
    return [ \
               tf.keras.callbacks.LearningRateScheduler(lambda x, lr: lr*lr_decay, verbose=1), \
               tf.keras.callbacks.EarlyStopping(patience=patience), \
               tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True), \
               ]

@ex.main
def main(n_process, max_queue_size, num_epochs, model_name, verbose):
    tf.keras.backend.clear_session()
    model = get_model()
    model.summary()
    # model.fit(get_train_dataset(), steps_per_epoch=100, epochs=1)
    history = model.fit( \
        get_train_dataset(), \
        validation_data=get_validation_dataset(), \
        validation_steps=get_validation_steps_per_epoch(), \
        steps_per_epoch=get_steps_per_epoch(), \
        epochs=num_epochs, \
        callbacks=get_callbacks(), \
        verbose=verbose)
    ex.add_artifact(model_name)
    bestModel = tf.keras.models.load_model(model_name, custom_objects={"f1":f1}, compile=True)
    test = get_test_dataset()
    testIterator = test.take(get_test_steps_per_epoch()).make_one_shot_iterator()
    all_ys_over_time = []
    predicted_ys_over_time = []
    all_ys_overall = []
    predicted_ys_overall = []
    for x, y in testIterator:
            over_time_y = y
            all_ys_over_time.append(over_time_y.numpy())
            predicted_y_over_time = bestModel.predict(x.numpy())
            predicted_ys_over_time.append(predicted_y_over_time)
            # all_ys_overall.append(overall_y)
            # predicted_ys_overall.append(predicted_y_overall)



    all_ys_over_time = np.vstack(all_ys_over_time)
    predicted_ys_over_time = np.vstack(predicted_ys_over_time)
    # all_ys_overall = np.vstack(all_ys_overall)
    # predicted_ys_overall = np.vstack(predicted_ys_overall)

    toReturn = {
        "history": history.history,
        "raw": {
            "predictions_over_time": predicted_ys_over_time,
            "labels_over_time": all_ys_overall,
            "predictions_overall": predicted_ys_overall,
            "labels_overall": all_ys_overall
        },
        "seizure_over_time": {
            "f1": f1_score(predictions_flattened, labels_flattened),
            "classification_report": classification_report(all_ys_over_time.reshape(-1,2).argmax(1), predicted_ys_over_time.reshape(-1,2).argmax(1), output_dict=True),
            "auc": roc_auc_score(all_ys_over_time.reshape(-1,2).argmax(1), predicted_ys_over_time.reshape(-1,2).argmax(1)),
            },
        # "seizure_overall": {
        #     "classification_report": classification_report(all_ys_overall.argmax(1), predicted_ys_overall.overall(1))
        # }
        }
    # raise Exception()
    return toReturn


if __name__ == "__main__":
    ex.run_commandline()
