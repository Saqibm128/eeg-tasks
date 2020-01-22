from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import clinical_text_analysis as cta
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
import data_reader as read
from keras import backend as K

# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import wf_analysis.datasets as wfdata
from keras_models.dataGen import EdfDataGenerator, DataGenMultipleLabels, RULEdfDataGenerator, RULDataGenMultipleLabels
from keras_models.cnn_models import vp_conv2d, conv2d_gridsearch, inception_like_pre_layers, conv2d_gridsearch_pre_layers
from keras import optimizers
from keras.layers import Dense, TimeDistributed, Input, Reshape, Dropout, LSTM, Flatten, Concatenate, CuDNNLSTM, GaussianNoise, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, TimeDistributed, Dense
import keras.layers as layers
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
import pickle as pkl
import sacred
import keras
import ensembleReader as er
from keras.utils import multi_gpu_model
from keras_models import train
import constants
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras_models.metrics import f1
import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import multi_gpu_model
from time import time
from ingredients.common import *
from addict import Dict
ex = sacred.Experiment(name="seizure_conv_exp_domain_adapt_v4", ingredients=[cnn_ingredient])

ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

# https://pynative.com/python-generate-random-string/
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@ex.named_config
def no_lin_pre_layer():
    num_lin_layer = 0

@ex.named_config
def no_stride_channels():
    '''
    Don't stride on channels
    '''
    max_pool_stride = (1,1)

@ex.named_config
def use_extra_layers():
    num_lin_layer = 2
    pre_layer_h = 22
    linear_dropout = 0.5
    num_post_cnn_layers = 2

@ex.named_config
def MLP():
    num_post_cnn_layers = 2
    linear_dropout=0.5
    num_post_lin_h=200


@ex.named_config
def knn():
    train_pkl = "/home/msaqib/train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/home/msaqib/valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/home/msaqib/test_multiple_labels_sessions_seizure_data_4.pkl"
    include_seizure_type = True
    session_instead_patient = True
    max_bckg_samps_per_file = None
    max_bckg_samps_per_file_test = None



@ex.named_config
def use_patient_dbmi():
    train_pkl = "/n/scratch2/ms994/train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_seizure_data_4.pkl"
    session_instead_patient = False

@ex.named_config
def debug_knn():
    train_pkl = "/home/msaqib/debug_train_multiple_labels_seizure_data_4.pkl"
    valid_pkl = "/home/msaqib/debug_valid_multiple_labels_seizure_data_4.pkl"
    test_pkl = "/home/msaqib/debug_test_multiple_labels_seizure_data_4.pkl"
    max_bckg_samps_per_file = 5 #limits number of samples we grab that are bckg to increase speed and reduce data size
    max_bckg_samps_per_file_test = 5
    max_samples=10000
    include_seizure_type=True
    session_instead_patient = True

@ex.named_config
def full_data():
    max_samples=None
    max_bckg_samps_per_file=None
    max_bckg_samps_per_file_test=None
    train_pkl = "/n/scratch2/ms994/full_train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/full_valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/test_multiple_labels_sessions_seizure_data_4.pkl"
    session_instead_patient = True
    include_seizure_type = True

@ex.named_config
def use_session_knn():
    train_pkl = "/home/msaqib/train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/home/msaqib/valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/home/msaqib/test_multiple_labels_sessions_seizure_data_4.pkl"
    session_instead_patient = True
    include_seizure_type = True
    max_bckg_samps_per_file = 100
    max_bckg_samps_per_file_test = -1


@ex.named_config
def most_common_seiz_types():
    seizure_classes_to_use=["bckg", "gnsz", "fnsz", "cpsz"]
    train_pkl = "/n/scratch2/ms994/gnfncp_train_multiple_labels_sessions_seizure_data_4.pkl"
    valid_pkl = "/n/scratch2/ms994/gnfncp_valid_multiple_labels_sessions_seizure_data_4.pkl"
    test_pkl = "/n/scratch2/ms994/gnfncp_test_multiple_labels_sessions_seizure_data_4.pkl"
    session_instead_patient = True
    max_bckg_samps_per_file_test=None
    include_seizure_type = True
    max_bckg_samps_per_file = 100
    max_bckg_samps_per_file_test = -1


@ex.named_config
def gnsz_fnsz():
    seizure_classes_to_use=["bckg", "gnsz", "fnsz"]

@ex.named_config
def gnsz():
    seizure_classes_to_use=["bckg", "gnsz"]

@ex.named_config
def fnsz():
    seizure_classes_to_use=["bckg", "fnsz"]

@ex.named_config
def measure_patient_bias():
    test_patient_model_after_train = True
    train_patient_model_after_train = True
    valid_patient_model_after_train = True

@ex.config
def config():
    steps_per_epoch = None


    imbalanced_resampler = "rul"
    change_batch_size_over_time = None

    fit_generator_verbosity = 2
    hyperopt_run = False
    validation_f1_score_type = None



    epochs=100

    test_patient_model_after_train = False
    train_patient_model_after_train = False
    valid_patient_model_after_train = False
    random_rearrange_each_batch = False
    random_rescale = False
    rescale_factor = 1.3


@ex.capture
def getImbResampler(imbalanced_resampler):
    if imbalanced_resampler is None:
        return None
    elif imbalanced_resampler == "SMOTE":
        return SMOTE()
    elif imbalanced_resampler == "rul":
        return RandomUnderSampler()


def get_random_channel_ordering():
    channel_ordering = [i for i in range(len(util_funcs.get_common_channel_names()))]
    np.random.shuffle(channel_ordering)
    return channel_ordering


@ex.capture
def resample_x_y(x, y, imbalanced_resampler):
    if imbalanced_resampler is None:
        return x, y
    else:
        oldShape = x.shape
        resampleX, resampleY = getImbResampler().fit_resample(x.reshape(x.shape[0], -1), y)
        return resampleX.reshape(resampleX.shape[0], *oldShape[1:]), resampleY



@ex.capture
def false_alarms_per_hour(fp, total_samps, num_seconds):
    num_chances_per_hour = 60 * 60 / num_seconds
    return (fp / total_samps) * num_chances_per_hour

@ex.capture
def get_test_patient_edg(test_pkl, batch_size):
    test_edss = pkl.load(open(test_pkl, "rb"))
    patients = [datum[1] for datum in test_edss]
    allPatients = list(set(patients))
    patientInd = [allPatients.index(patient) for patient in patients]
    num_patients = len(allPatients)
    # x_data = [datum[0] for datum in test_edss]
    test_edg = EdfDataGenerator(test_edss, labels=patientInd, n_classes=num_patients, batch_size=batch_size, shuffle=True, precache=True)
    return test_edg, num_patients

@ex.capture
def train_patient_accuracy_after_training(x_input, cnn_y, trained_model, train_pkl):
    return test_patient_accuracy_after_training(x_input, cnn_y, trained_model, test_pkl=train_pkl)

@ex.capture
def valid_patient_accuracy_after_training(x_input, cnn_y, trained_model, valid_pkl):
    return test_patient_accuracy_after_training(x_input, cnn_y, trained_model, test_pkl=valid_pkl)



@ex.capture
def test_patient_accuracy_after_training(x_input, cnn_y, trained_model, lr, lr_decay, epochs, model_name, fit_generator_verbosity, test_pkl, num_patients=None):
    # if test_edg is None:
    test_edg, num_patients = get_test_patient_edg(test_pkl=test_pkl)
    # train_test_edg, valid_test_edg = test_edg.create_validation_train_split()
    patient_layer = Dense(num_patients, activation="softmax")(cnn_y)
    patient_model = Model(inputs=[x_input], outputs=[patient_layer])
    for i, layer in enumerate(patient_model.layers[:-1]):
        layer.set_weights(trained_model.layers[i].get_weights())
        layer.trainable = False #freeze all layers except last
    print(patient_model.summary())
    patient_model.compile(get_optimizer()(lr=lr), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
    test_patient_history = patient_model.fit_generator(test_edg, epochs=epochs, verbose=2, callbacks=[get_model_checkpoint(model_name[:-3] + "_patient.h5"), get_early_stopping(early_stopping_on="loss"), LearningRateScheduler(lambda x, old_lr: old_lr * lr_decay) ])
    return test_patient_history

@ex.main
def main(model_name, mode, num_seconds, imbalanced_resampler,  regenerate_data, epochs, fit_generator_verbosity, batch_size, n_process, steps_per_epoch, patience,
         include_seizure_type, max_bckg_samps_per_file_test, seizure_weight, seizure_weight_decay, update_seizure_class_weights, seizure_classification_only,
         validation_f1_score_type, reduce_lr_on_plateau, lr, lr_decay, change_batch_size_over_time,
         test_patient_model_after_train, train_patient_model_after_train, valid_patient_model_after_train,
         random_rearrange_each_batch, random_rescale, rescale_factor, include_montage_channels):
    seizure_class_weights = {0:1,1:1}
    edg, valid_edg, test_edg, len_all_patients = get_data_generators()
    # patient_class_weights = {}
    # for i in range(len_all_patients):
    #     patient_class_weights[i] = 1

    print("Creating models")
    seizure_model, seizure_patient_model, patient_model, val_train_model, x_input, cnn_y, loss_weights = get_model(num_patients=len_all_patients)

    if regenerate_data:
        return

    # if steps_per_epoch is None:
    #     history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs)
    # else:
    #     history = model.fit_generator(edg, validation_data=valid_edg, callbacks=get_cb_list(), verbose=fit_generator_verbosity, epochs=epochs, steps_per_epoch=steps_per_epoch)


    # train_ordered_enqueuer = OrderedEnqueuer(edg, True)
    # valid_ordered_enqueuer = OrderedEnqueuer(valid_edg, True)


    num_epochs = epochs
    training_seizure_accs = []
    valid_seizure_accs = []
    train_patient_accs = []
    training_seizure_loss = []
    train_seizure_f1s = []
    train_patient_f1s = []
    train_subtype_f1s = []
    train_montage_f1s = []
    valid_seizure_loss = []
    valid_f1_scores = []
    train_montage_loss = []
    train_montage_acc = []
    val_montage_loss = []
    val_montage_acc = []

    oldPatientWeights = patient_model.layers[-1].get_weights()
    oldNonPatientWeights = [layer.get_weights() for layer in seizure_model.layers[:-1]]
    best_model_loss = -100
    patience_left = patience
    if include_seizure_type:
        subtype_accs = []
        subtype_losses = []
        valid_seizure_subtype_accs = []
        valid_seizure_subtype_loss = []
    if reduce_lr_on_plateau:
        lrs = []
        current_lr = lr
    if change_batch_size_over_time is not None:
        batch_sizes = []
        current_batch_size = edg.batch_size
        # seizure_weights = []
        # current_seizure_weight = seizure_weight

    for i in range(num_epochs):
        if patience_left == 0:
            continue



        if reduce_lr_on_plateau:
            lrs.append(current_lr)
            # seizure_weights.append(current_seizure_weight)
            recompile_model(seizure_patient_model, i, loss_weights=loss_weights, lr=current_lr)
        else:
            recompile_model(seizure_patient_model, i, loss_weights=loss_weights,)
        if change_batch_size_over_time is not None:
            batch_sizes.append(current_batch_size)


        valid_labels_full_epoch = []
        valid_labels_epoch= []
        valid_predictions_full = []
        valid_predictions = []

        if include_montage_channels:
            montage_epochs_accs = []
            montage_val_epoch_labels = []
            montage_val_predictions_epoch = []
            montage_val_epoch_labels_full = []
            montage_val_predictions_epoch_full = []

        if include_seizure_type:
            subtype_epochs_accs = []
            subtype_val_epoch_labels = []
            subtype_val_predictions_epoch = []
            subtype_val_epoch_labels_full = []
            subtype_val_predictions_epoch_full = []





        train_seizure_loss_epoch = []
        train_subtype_loss_epoch = []
        train_montage_loss_epoch = []
        train_seizure_f1_epoch = []
        train_subtype_f1_epoch = []
        train_patient_f1_epoch = []
        train_montage_f1_epoch = []

        seizure_accs = []
        patient_accs_epoch = []
        train_montage_acc_epoch = []
        # for j in range(len(edg)):
        if steps_per_epoch is None:
            steps_per_epoch_func = lambda: len(edg)
        else:
            steps_per_epoch_func = lambda: steps_per_epoch
        for j in range(steps_per_epoch_func()):

            train_batch = edg[j]
            data_x = train_batch[0]
            data_x = data_x.astype(np.float32)
            data_x = np.nan_to_num(data_x)

            if random_rearrange_each_batch:
                data_x = data_x[:,:,np.random.choice(21, 21, replace=False)]

            if random_rescale:
                data_x = data_x * (np.random.random() * (rescale_factor - 1/rescale_factor) + 1/rescale_factor)

            if include_seizure_type and include_montage_channels:
                loss, seizure_loss, patient_loss, subtype_loss, montage_loss, seizure_acc, seizure_f1, patient_acc, patient_f1,  subtype_acc, subtype_f1, montage_acc, montage_f1 = seizure_patient_model.train_on_batch(data_x, train_batch[1], )
                subtype_epochs_accs.append(subtype_acc)
                # raise Exception()
                train_subtype_f1_epoch.append(subtype_f1)
                train_montage_f1_epoch.append(montage_f1)
            elif include_seizure_type:
                loss, seizure_loss, patient_loss, subtype_loss, seizure_acc, seizure_f1, patient_acc, patient_f1, subtype_acc, subtype_f1 = seizure_patient_model.train_on_batch(data_x, train_batch[1], )
                subtype_epochs_accs.append(subtype_acc)
                train_subtype_f1_epoch.append(subtype_f1)
            elif not include_seizure_type and not include_montage_channels:
                loss, seizure_loss, patient_loss, seizure_acc, seizure_f1, patient_acc, patient_f1 = seizure_patient_model.train_on_batch(data_x, train_batch[1])
            seizure_accs.append(seizure_acc)
            train_seizure_f1_epoch.append(seizure_f1)
            train_patient_f1_epoch.append(patient_f1)


            #old patient weights are trying to predict for patient, try to do the prediction!
            patient_model.layers[-1].set_weights(oldPatientWeights)
            #keep the other nonpatient weights which try not to predict for patient!
            oldNonPatientWeights = [layer.get_weights() for layer in seizure_model.layers[:-1]]
            patient_loss, patient_acc = patient_model.train_on_batch(train_batch[0], train_batch[1][1])
            patient_accs_epoch.append(patient_acc)

            train_seizure_loss_epoch.append(seizure_loss)
            if include_seizure_type:
                train_subtype_loss_epoch.append(subtype_loss)
            if include_montage_channels:
                train_montage_loss_epoch.append(montage_loss)
                train_montage_acc_epoch.append(montage_acc)

            #get weights that try to predict for patient
            oldPatientWeights = patient_model.layers[-1].get_weights()

            #set weights that don't ruin seizure prediction
            for layer_num, layer in enumerate(seizure_model.layers[:-1]):
                seizure_model.layers[layer_num].set_weights(oldNonPatientWeights[layer_num])
            if (j % int(len(edg)/10)) == 0:
                printEpochUpdateString = "epoch: {} batch: {}/{}, seizure acc: {}, seizure f1: {}, patient acc: {}, loss: {}".format(i, j, len(edg), np.mean(seizure_accs), np.mean(train_seizure_f1_epoch), np.mean(patient_accs_epoch), loss)
                if include_seizure_type:
                    printEpochUpdateString += ", seizure subtype acc: {}, subtype loss: {}".format(np.mean(subtype_epochs_accs), np.mean(train_subtype_loss_epoch))
                if include_montage_channels:
                    printEpochUpdateString += ", seizure montage identification acc: {}, montage loss: {}".format(np.mean(train_montage_acc_epoch), np.mean(train_montage_loss_epoch))
                print(printEpochUpdateString)
    #     valid_edg.start_background()

        assert valid_labels_epoch == []
        assert valid_predictions == []


        for j in range(len(valid_edg)):
            valid_batch = valid_edg[j]
            data_x = valid_batch[0]
            data_x = data_x.astype(np.float32)
            data_x = np.nan_to_num(data_x) #ssome weird issue with incorrect data conversion


            val_batch_predictions = val_train_model.predict_on_batch(data_x)
            if include_montage_channels and include_seizure_type:
                # montage_val_predictions_epoch.append(val_batch_predictions[2].argmax(1))
                montage_val_predictions_epoch_full.append(val_batch_predictions[2])
                # montage_val_epoch_labels.append(valid_batch[1][3].argmax(1))
                montage_val_epoch_labels_full.append(valid_batch[1][3])

            if include_seizure_type:
                subtype_val_predictions_epoch.append(val_batch_predictions[1].argmax(1))
                subtype_val_predictions_epoch_full.append(val_batch_predictions[1])
                subtype_val_epoch_labels.append(valid_batch[1][2].argmax(1))
                subtype_val_epoch_labels_full.append(valid_batch[1][2])
                valid_labels_epoch.append(valid_batch[1][0].argmax(1))
                valid_labels_full_epoch.append(valid_batch[1][0])
                valid_predictions.append(val_batch_predictions[0].argmax(1))
                valid_predictions_full.append(val_batch_predictions[0])
            else:
                valid_labels_epoch.append(valid_batch[1][0].argmax(1))
                valid_labels_full_epoch.append(valid_batch[1][0])
                valid_predictions.append(val_batch_predictions.argmax(1))
                valid_predictions_full.append(val_batch_predictions)

        def get_sum_seizures():
            num_seizures = 0
            for j in range(len(valid_edg)):
                valid_batch = valid_edg[j]
                num_seizures += valid_batch[1][0].argmax(1).sum()
            return num_seizures

        #random infinitye predictions? I'm assuming some weird type conversion issues and that nan_to_num should fix this

        valid_labels_epoch= np.nan_to_num(np.hstack(valid_labels_epoch).astype(np.float32))
        valid_predictions = np.nan_to_num(np.hstack(valid_predictions).astype(np.float32))

        print("debug: valid_labels_epoch shape {}, valid_predictions.shape {}".format(valid_labels_epoch.shape, valid_predictions.shape))
        print("We predicted {} seizures in the validation split, there were actually {}".format(valid_predictions.sum(), valid_labels_epoch.sum()))
        print("We predicted {} seizure/total in the validation split, there were actually {}".format(valid_predictions.sum()/len(valid_predictions), valid_labels_epoch.sum()/len(valid_labels_epoch)))
        print(classification_report(valid_labels_epoch, valid_predictions))

        if update_seizure_class_weights and valid_predictions.sum()/len(valid_predictions) > 0.95:
            seizure_class_weights[0] *= 1.05
            seizure_class_weights[1] /= 1.05
            print("Updating seizure classes {}".format(seizure_class_weights))
        elif update_seizure_class_weights and valid_predictions.sum()/len(valid_predictions) < 0.05:
            seizure_class_weights[1] *= 1.05
            seizure_class_weights[0] /= 1.05
            print("Updating seizure classes {}".format(seizure_class_weights))




        valid_labels_full_epoch = np.nan_to_num(np.vstack(valid_labels_full_epoch).astype(np.float32))
        valid_predictions_full = np.nan_to_num(np.vstack(valid_predictions_full).astype(np.float32))

        if include_montage_channels:
            # montage_val_epoch_labels = np.nan_to_num(np.hstack(montage_val_epoch_labels).astype(np.float32))
            # montage_val_predictions_epoch = np.nan_to_num(np.hstack(montage_val_predictions_epoch).astype(np.float32))
            montage_val_epoch_labels_full = np.nan_to_num(np.vstack(montage_val_epoch_labels_full).astype(np.float32))
            montage_val_predictions_epoch_full = np.nan_to_num(np.vstack(montage_val_predictions_epoch_full).astype(np.float32))

        if include_seizure_type:
            subtype_val_epoch_labels = np.nan_to_num(np.hstack(subtype_val_epoch_labels).astype(np.float32))
            subtype_val_predictions_epoch = np.nan_to_num(np.hstack(subtype_val_predictions_epoch).astype(np.float32))
            subtype_val_epoch_labels_full = np.nan_to_num(np.vstack(subtype_val_epoch_labels_full).astype(np.float32))
            subtype_val_predictions_epoch_full = np.nan_to_num(np.vstack(subtype_val_predictions_epoch_full).astype(np.float32))



        try:
            auc = roc_auc_score(valid_predictions, valid_labels_epoch)
        except Exception:
            auc = "undefined"
        valid_acc =  accuracy_score(valid_predictions, valid_labels_epoch)
        valid_seizure_accs.append(valid_acc)
        train_patient_accs.append(np.mean(patient_accs_epoch))
        valid_loss = log_loss(valid_labels_full_epoch, valid_predictions_full)
        training_seizure_loss.append(np.mean(train_seizure_loss_epoch))
        train_seizure_f1s.append(np.mean(train_seizure_f1_epoch))
        train_patient_f1s.append(np.mean(train_patient_f1_epoch))

        printEpochEndString = "end epoch: {}, f1: {}, auc: {}, acc: {}, loss: {}\n".format(i, f1_score(valid_predictions, valid_labels_epoch), auc, valid_acc, valid_loss)
        valid_f1_scores.append(f1_score(valid_predictions, valid_labels_epoch))
        valid_seizure_loss.append(valid_loss)
        if include_montage_channels:
            train_montage_f1s.append(np.mean(train_montage_f1_epoch))
            train_montage_loss.append(np.mean(train_montage_loss_epoch))
            train_montage_acc.append(np.mean(train_montage_acc_epoch))
            current_val_epoch_montage_acc = accuracy_score(montage_val_epoch_labels_full, np.round(montage_val_predictions_epoch_full).astype(np.int))
            current_val_epoch_montage_loss = log_loss(montage_val_epoch_labels_full, montage_val_predictions_epoch_full)
            val_montage_acc.append(current_val_epoch_montage_acc)
            val_montage_loss.append(current_val_epoch_montage_loss)
            printEpochEndString += "\t montage info: train acc: {}, train f1: {}, valid acc:{}, loss: {}\n".format(train_montage_acc[-1], train_seizure_f1s[-1], val_montage_acc[-1], val_montage_loss[-1],)

        if include_seizure_type:
            train_subtype_f1s.append(np.mean(train_subtype_f1_epoch))
            subtype_losses.append(np.mean(train_subtype_loss_epoch))
            subtype_acc = np.mean(subtype_epochs_accs)
            subtype_accs.append(subtype_acc)
            val_subtype_acc = accuracy_score(subtype_val_epoch_labels, subtype_val_predictions_epoch)
            valid_seizure_subtype_accs.append(val_subtype_acc)
            val_subtype_loss = log_loss(subtype_val_epoch_labels_full, subtype_val_predictions_epoch_full)
            valid_seizure_subtype_loss.append(val_subtype_loss)
            macro_subtype_f1 = f1_score(subtype_val_epoch_labels, subtype_val_predictions_epoch, average='macro')
            weighted_subtype_f1 = f1_score(subtype_val_epoch_labels, subtype_val_predictions_epoch, average='weighted')
            printEpochEndString += "\tsubtype info: train acc: {}, valid acc:{}, loss: {}, macro_f1: {}, weighted_f1: {}\n\n".format(subtype_acc, val_subtype_acc, val_subtype_loss, macro_subtype_f1, weighted_subtype_f1)



        print(printEpochEndString)

        if seizure_classification_only:
            new_val_f1 = weighted_subtype_f1
        elif validation_f1_score_type is None:
            new_val_f1 = f1_score(valid_predictions, valid_labels_epoch)
        else:
            new_val_f1 = f1_score(valid_predictions, valid_labels_epoch, average=validation_f1_score_type)
        if (new_val_f1 > best_model_loss):
            patience_left = patience
            best_model_loss = new_val_f1
            try:
                val_train_model.save(model_name)
                print("improved val score to {}".format(best_model_loss))
            except Exception as e:
                print("{}\n".format(e))
                print("failed saving\n")
        else:
            patience_left -= 1
            if reduce_lr_on_plateau:
                current_lr = current_lr * lr_decay
                print("changing batch size {}".format(current_batch_size))

            if patience_left == 0:
                print("Early Stopping!")
        if change_batch_size_over_time is not None:
            edg.batch_size = max(int(edg.batch_size * 3/4), change_batch_size_over_time)
            current_batch_size=edg.batch_size




        training_seizure_accs.append(np.mean(seizure_accs))

        edg.on_epoch_end()
        # valid_edg.on_epoch_end()

    del edg
    del valid_edg
    model = load_model(model_name)




    y_pred = model.predict_generator(test_edg)


    results = Dict()
    results.history = Dict({
        "binary_accuracy": training_seizure_accs,
        "val_binary_accuracy": valid_seizure_accs,
        "seizure_loss": training_seizure_loss,
        "valid_seizure_loss": valid_seizure_loss,
        "patient_acc": train_patient_accs,

    })
    if reduce_lr_on_plateau:
        results.history.lr = lrs
    if change_batch_size_over_time:
        results.history.batch_size = batch_sizes
    if train_patient_model_after_train:
        print("train patient measurement")
        results.patient_history.train = train_patient_accuracy_after_training(x_input, cnn_y, model).history
    if valid_patient_model_after_train:
        print("valid patient measurement")
        results.patient_history.valid = valid_patient_accuracy_after_training(x_input, cnn_y, model).history
    if test_patient_model_after_train:
        print("test patient measurement")
        test_patient_history = test_patient_accuracy_after_training(x_input, cnn_y, model)
        results.patient_history.test = test_patient_history.history


    results.history.seizure.valid_f1 = valid_f1_scores
    results.history.seizure.train_f1 = train_seizure_f1s


    if include_seizure_type:
        results.history.subtype.train_f1 = train_seizure_f1s
        results.history.subtype.acc = subtype_accs
        results.history.subtype.val_acc = valid_seizure_subtype_accs
        results.history.subtype.loss = subtype_losses
        results.history.subtype.val_loss = valid_seizure_subtype_loss

    if include_montage_channels:
        results.history.montage.train_f1 = train_montage_f1s
        results.history.montage.train_acc = train_montage_acc
        results.history.montage.train_loss = train_montage_loss
        results.history.montage.val_acc = val_montage_acc
        results.history.montage.val_loss = val_montage_loss

    if include_seizure_type:
        y_seizure_label =  np.array([data[1][0] for data in test_edg.dataset]).astype(int)
        y_seizure_pred = np.array([y_pred[0].argmax(1)]).astype(int)[0]
        y_subtype_label =  np.array([data[1][2] for data in test_edg.dataset]).astype(int)
        y_subtype_pred = np.array([y_pred[1].argmax(1)]).astype(int)[0]
        results.subtype.acc = accuracy_score(y_subtype_label, y_subtype_pred)
        results.subtype.f1.macro = f1_score(y_subtype_label, y_subtype_pred, average='macro')
        results.subtype.f1.micro = f1_score(y_subtype_label, y_subtype_pred, average='micro')
        results.subtype.f1.weighted = f1_score(y_subtype_label, y_subtype_pred, average='weighted')
        results.subtype.confusion_matrix = confusion_matrix(y_subtype_pred, y_subtype_label)
        results.subtype.classification_report = classification_report(y_subtype_pred, y_subtype_label, output_dict=True)
    else:
        y_seizure_label =  np.array([data[1] for data in test_edg.dataset]).astype(int)
        y_seizure_pred = np.array(y_pred.argmax(1)).astype(int)

    if include_seizure_type and include_montage_channels:
        y_montage_label =  np.array([data[1][3] for data in test_edg.dataset]).astype(int)
        y_montage_pred = np.round(y_pred[2]).astype(int)
        results.montage.acc = accuracy_score(y_montage_label, y_montage_pred)
        results.montage.f1.macro = f1_score(y_montage_label, y_montage_pred, average='macro')
        results.montage.f1.micro = f1_score(y_montage_label, y_montage_pred, average='micro')
        results.montage.f1.weighted = f1_score(y_montage_label, y_montage_pred, average='weighted')
        # results.montage.confusion_matrix = confusion_matrix(y_montage_pred, y_montage_label)
        # results.montage.classification_report = classification_report(y_montage_pred, y_montage_label, output_dict=True)

    print("We predicted {} seizures in the test split, there were actually {}".format(y_seizure_pred.sum(), np.array([data[1][0] for data in test_edg.dataset]).astype(int).sum()))
    print("We predicted {} seizure/total in the test split, there were actually {}".format(y_seizure_pred.sum()/len(y_seizure_pred), np.array([data[1][0] for data in test_edg.dataset]).astype(int).sum()/len(test_edg.dataset)))

    if not seizure_classification_only:
        results.seizure.acc = accuracy_score(y_seizure_pred, y_seizure_label)
        results.seizure.f1 = f1_score(y_seizure_pred, y_seizure_label)
        results.seizure.classification_report = classification_report(y_seizure_label, y_seizure_pred, output_dict=True)
        results.seizure.confusion_matrix = confusion_matrix(y_seizure_label, y_seizure_pred)
        if max_bckg_samps_per_file_test is not None or max_bckg_samps_per_file_test==-1:
            total_samps = sum(results.seizure.confusion_matrix[0]) #just use the samps labeled negative, max_bckg_samps_per_file_test is used to run faster but leads to issues with class imbalance not being fully reflected if we include seizure
        else:
            total_samps = sum(sum(results.seizure.confusion_matrix))
        results.seizure.false_alarms_per_hour = false_alarms_per_hour(results.seizure.confusion_matrix[0][1], total_samps=total_samps)

        try:
            results.seizure.AUC = roc_auc_score(y_seizure_pred, y_seizure_label)
        except Exception:
            results.seizure.AUC = "failed to calculate"

    return results.to_dict()


if __name__ == "__main__":
    ex.run_commandline()
