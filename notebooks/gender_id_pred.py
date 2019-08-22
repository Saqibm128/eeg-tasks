#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.path.realpath(".."))
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"

import util_funcs
from importlib import reload
import data_reader as read
import ensembleReader as er
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import constants
import clinical_text_analysis as cta
import tsfresh
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from os import path
import keras_models.dataGen as dg
import keras_models.cnn_models as cnn_models
reload(dg)
import predictGenderConvExp as pg
from addict import Dict
from sklearn.model_selection import train_test_split


# In[2]:


import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Concatenate, Dropout, MaxPool2D, Conv3D, Flatten, LeakyReLU, BatchNormalization


# In[3]:


files, genders = cta.demux_to_tokens(cta.getGenderAndFileNames("combined", "01_tcp_ar", True))


# In[4]:


sessions = [read.parse_edf_token_path_structure(file)[2] for file in files]
patients = [read.parse_edf_token_path_structure(file)[1] for file in files]


# In[5]:


allPatients = list(set(patients))


# In[6]:


allPatients.sort()


# In[7]:


len(allPatients)


# In[8]:


sessionDict = Dict()
for i, file in enumerate(files):
    sessionDict[patients[i]][sessions[i]][i].file = file
    sessionDict[patients[i]][sessions[i]][i].gender = genders[i]
    sessionDict[patients[i]][sessions[i]][i].patient = allPatients.index(patients[i])


# In[9]:


#delete all patients with only one session
for patient in set(patients):
    print(len(sessionDict[patient].keys()))
    if len(sessionDict[patient].keys()) < 2:
        del sessionDict[patient]


# In[10]:


def returnFilesAndLabelsFromSessionDict(d):
    files = []
    genders = []
    for id_num in d.keys():
        files.append(d[id_num].file)
        genders.append((d[id_num].gender, d[id_num].patient))
    return files, genders


# In[11]:


testPatientFiles = []
testLabels = []
trainPatientFiles = []
trainLabels = []
for patient in sessionDict.keys():
    testSessionToAdd = np.random.choice(list(sessionDict[patient].keys()))
    for session in sessionDict[patient].keys():
        files, genders = returnFilesAndLabelsFromSessionDict(sessionDict[patient][session])
        if session == testSessionToAdd:
            testPatientFiles += files
            testLabels += genders
        else:
            trainPatientFiles += files
            trainLabels += genders


# In[12]:


len(testPatientFiles), len(trainPatientFiles)


# In[13]:


len(allPatients)


# In[14]:


len(genders)


# In[15]:


reload(er)
testEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=5, edf_tokens=testPatientFiles, n_process=4, labels=testLabels)[:]
trainEnsembler = er.EdfDatasetEnsembler("combined", "01_tcp_ar", max_num_samples=5, edf_tokens=trainPatientFiles, n_process=4, labels=trainLabels)[:]


# In[16]:


len(trainEnsembler)


# In[17]:


trainEnsembler, validEnsembler = train_test_split(trainEnsembler, test_size=0.1)


# In[ ]:





# In[18]:


reload(dg)
testDataGen = dg.DataGenMultipleLabels(testEnsembler, batch_size=128, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False)
validDataGen = dg.DataGenMultipleLabels(validEnsembler, batch_size=128, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False)
trainDataGen = dg.DataGenMultipleLabels(trainEnsembler, batch_size=64, num_labels=2, n_classes=(2, len(allPatients)), precache=True, time_first=False, shuffle=False)


# In[19]:


len(testDataGen), len(trainDataGen)


# In[20]:


x, y = testDataGen[0]


# In[21]:


x.shape


# In[22]:


y


# In[108]:


x, y = cnn_models.inception_like_pre_layers(input_shape=(21, 500, 1), num_filters=20, dropout=0.5, num_layers=4)


# In[109]:


y_gender = Dense(2, activation="softmax", name="gender")(y)
y_patient = Dense(len(allPatients), activation="softmax", name="patient")(y)


# In[110]:


model = Model(inputs=x, outputs=[y_gender, y_patient])


# In[111]:


model.summary()
model.save("baseModel.h5")


# In[112]:


from keras.utils import multi_gpu_model
model = multi_gpu_model(model, 2)


# In[113]:


adam = keras.optimizers.Adam(lr=0.0005)
model_gender_save = keras.callbacks.ModelCheckpoint("genderPredictor.h5", monitor="val_gender_loss", save_best_only=True, mode="min", verbose=True)
model_patient_save = keras.callbacks.ModelCheckpoint("patientPredictor.h5", monitor="val_patient_loss", save_best_only=True, mode="min", verbose=True)


earlyStopsOnGender = keras.callbacks.EarlyStopping(monitor="val_gender_loss", mode="min", patience=5, verbose=True)
earlyStopsOnPatient = keras.callbacks.EarlyStopping(monitor="val_patient_loss", mode="min", patience=5, verbose=True)
cb = [model_gender_save, model_patient_save, earlyStopsOnPatient]

model.compile(adam, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=["categorical_accuracy"], loss_weights=[0.5, 1.5])


# In[114]:


model.summary()


# In[115]:

#
# history = model.fit_generator(trainDataGen, epochs=100, validation_data=validDataGen, callbacks=cb)
#
#
# # In[116]:
#
#
# genderPredictor = keras.models.load_model("genderPredictor.h5")
# genderPredictor = multi_gpu_model(genderPredictor, 4)
#
#
# # In[117]:
#
#
# len(testEnsembler)
#
#
# # In[118]:
#
#
# ypred = genderPredictor.predict(np.stack([data[0].reshape(500,21,1).transpose(1,0,2) for data in testEnsembler]))
#
#
# # In[119]:
#
#
# genderPred = ypred[0]
#
#
# # In[120]:
#
#
# accuracy_score(genderPred.argmax(1), [data[1][0] for data in testEnsembler])
#
#
# # In[121]:
#
#
# accuracy_score(ypred[1].argmax(1), [data[1][1] for data in testEnsembler])
#
#
# # In[122]:
#
#
# roc_auc_score(genderPred.argmax(1), [data[1][0] for data in testEnsembler])
#

# In[123]:


class_weights = [(0,1),(1,100), (1,75), (1,50), (1,25), (1, 10), (1,5), (1,1)]
class_weights += [(weight[1], weight[0]) for weight in class_weights]
class_weights = class_weights[:-1]
class_weights
# class_weights = list(set(class_weights))


# In[ ]:


results = Dict()
for class_weight in class_weights:
    print("starting {}".format(class_weight))
    model = keras.models.load_model("baseModel.h5")
    model = multi_gpu_model(model, 2)
    adam = keras.optimizers.Adam(lr=0.001)
    model_save = keras.callbacks.ModelCheckpoint("lossPredictor{}.h5".format(class_weight), monitor="val_loss", save_best_only=True, mode="min", verbose=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
    cb = [model_save, early_stopping]
    model.compile(adam, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=["categorical_accuracy"], loss_weights=list(class_weight))
    history = model.fit_generator(trainDataGen, epochs=100, validation_data=validDataGen, callbacks=cb, verbose=2)
    results[class_weight].history = history.history
    model = keras.models.load_model("lossPredictor{}.h5".format(class_weight))
    model = multi_gpu_model(model, 2)
    ypred = model.predict(np.stack([data[0].reshape(500,21,1).transpose(1,0,2) for data in testEnsembler]))
    results[class_weight].gender.acc = accuracy_score(ypred[0].argmax(1), [data[1][0] for data in testEnsembler])
    results[class_weight].gender.AUC = roc_auc_score(ypred[0].argmax(1), [data[1][0] for data in testEnsembler])
    results[class_weight].patient.acc = accuracy_score(ypred[1].argmax(1), [data[1][1] for data in testEnsembler])
    print(results[class_weight])
#     results[class_weight].patient.AUC = roc_auc_score(ypred[1].argmax(1), [data[1][1] for data in testEnsembler])
import pickle as pkl
pkl.dump(results, open("results.pkl", 'wb'))


# In[133]:
