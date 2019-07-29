import sys, os
sys.path.append(os.path.realpath(".."))
# os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit" #was using this for jupyter notebook to force gpu support





import util_funcs
from importlib import reload
import data_reader as read
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
import predictGenderConvExp as pg
from keras_models.vanPutten import inception_like

import sacred
from sacred.observers import MongoObserver
ex = sacred.Experiment(name="inception")
# ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))



import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation, Conv2D, Concatenate, Dropout, MaxPool2D, Conv3D, Flatten, LeakyReLU, BatchNormalization


# # importing in the preprocessed data


@ex.automain
def main():
    trainData = pkl.load(open("standardized_combined_simple_ensemble_train_data.pkl", 'rb'))

    testData = pkl.load(open("standardized_combined_simple_ensemble_test_data.pkl", 'rb'))




    validData = pkl.load(open("valid_standardized_combined_simple_ensemble_train_data.pkl", 'rb'))


    def generate_x_y(data):
        x_data = np.stack([datum[0] for datum in data])
        x_data = x_data.reshape((*x_data.shape, 1))
        x_data.transpose(0, 2, 1, 3)
        y_data = np.array([datum[1] for datum in data])
        y_data = keras.utils.to_categorical(y_data)
        return x_data, y_data

    testDataX, testDataY = generate_x_y(testData)
    del testData #freeing up memory so jupyter wouldn't crash

    trainDataX, trainDataY = generate_x_y(trainData)
    del trainData

    validDataX, validDataY = generate_x_y(validData)
    del validData


    dropout = 0.5

    x = Input(((500, 21, 1)))
    # y1 = Conv2D(20, (3,3),  activation="relu",)(x)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = Conv2D(40, (3,3), activation="relu",)(y1)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = Conv2D(40, (3,3), activation="relu",)(y1)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = Conv2D(40, (3,3), activation="relu",)(y1)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = Conv2D(40, (3,3), activation="relu",)(y1)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = Conv2D(40, (3,3), activation="relu",)(y1)
    # y1 = MaxPool2D(pool_size=(2, 1),)(y1)
    # y1 = Dropout(dropout)(y1)
    # y1 = Flatten()(y1)


    # In[12]:


    y0 = Conv2D(20, (2,2),  activation="relu",)(x)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    y0 = Conv2D(40, (2,2), activation="relu",)(y0)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    y0 = Conv2D(40, (2,2), activation="relu",)(y0)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    y0 = Conv2D(40, (2,2), activation="relu",)(y0)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    y0 = Conv2D(40, (2,2), activation="relu",)(y0)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = BatchNormalization()(y0)
    y0 = Conv2D(40, (2,2), activation="relu",)(y0)
    y0 = MaxPool2D(pool_size=(2, 1),)(y0)
    y0 = Dropout(dropout)(y0)
    y0 = Flatten()(y0)


    # In[13]:


    # y2 = Conv2D(20, (4,4),  activation="relu",)(x)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Conv2D(40, (4,4), activation="relu",)(y2)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Conv2D(40, (4,4), activation="relu",)(y2)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Conv2D(40, (4,4), activation="relu",)(y2)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Conv2D(40, (4,4), activation="relu",)(y2)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Conv2D(40, (4,4), activation="relu",)(y2)
    # y2 = MaxPool2D(pool_size=(2, 1),)(y2)
    # y2 = Dropout(dropout)(y2)
    # y2 = Flatten()(y2)


    # In[14]:

    #
    # y3 = Conv2D(20, (5,5),  activation="relu",)(x)
    # y3 = MaxPool2D(pool_size=(2, 1),)(y3)
    # y3 = Dropout(dropout)(y3)
    # y3 = BatchNormalization()(y3)
    # y3 = Conv2D(40, (5,5), activation="relu",)(y3)
    # y3 = MaxPool2D(pool_size=(2, 1),)(y3)
    # y3 = Dropout(dropout)(y3)
    # y3 = BatchNormalization()(y3)
    # y3 = Conv2D(40, (5,5), activation="relu",)(y3)
    # y3 = MaxPool2D(pool_size=(2, 1),)(y3)
    # y3 = Dropout(dropout)(y3)
    # y3 = BatchNormalization()(y3)
    # y3 = Conv2D(40, (5,5), activation="relu",)(y3)
    # y3 = MaxPool2D(pool_size=(2, 1),)(y3)
    # y3 = Dropout(dropout)(y3)
    # y3 = BatchNormalization()(y3)
    # y3 = Conv2D(40, (5,5), activation="relu",)(y3)
    # y3 = MaxPool2D(pool_size=(2, 1),)(y3)
    # y3 = Dropout(dropout)(y3)
    # y3 = Flatten()(y3)


    # In[16]:


    # y4 = Conv2D(20, (6,6),  activation="relu",)(x)
    # y4 = MaxPool2D(pool_size=(2, 1),)(y4)
    # y4 = Dropout(dropout)(y4)
    # y4 = BatchNormalization()(y4)
    # y4 = Conv2D(40, (6,6), activation="relu",)(y4)
    # y4 = MaxPool2D(pool_size=(2, 1),)(y4)
    # y4 = Dropout(dropout)(y4)
    # y4 = BatchNormalization()(y4)
    # y4 = Conv2D(40, (6,6), activation="relu",)(y4)
    # y4 = MaxPool2D(pool_size=(2, 1),)(y4)
    # y4 = Dropout(dropout)(y4)
    # y4 = BatchNormalization()(y4)
    # y4 = Conv2D(40, (6,6), activation="relu",)(y4)
    # y4 = MaxPool2D(pool_size=(2, 1),)(y4)
    # y4 = Dropout(dropout)(y4)
    # y4 = BatchNormalization()(y4)
    # y4 = Flatten()(y4)


    # In[17]:


    # y = Concatenate()([y0])

    y = Dense(units=2, activation="softmax")(y0)
    model = Model(inputs=x, outputs =y)


    # In[24]:

    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, 2)

    from keras.optimizers import Adam
    adam = Adam(lr=0.002)
    model.compile(adam, loss="categorical_crossentropy", metrics=["acc"])

    model.summary()

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    cb = [EarlyStopping(patience=10, verbose=True), ModelCheckpoint("yolo.h5",save_best_only=True, verbose=True)]


    # In[ ]:


    72490*500*21*8


    # In[ ]:


    sys.getsizeof(trainDataX)


    # In[ ]:


    trainDataX.shape


    # In[ ]:


    reload(dg)


    # In[22]:





    # In[ ]:


    step_data_size=500
    for i in range(1):
        for j in range(70*2):
            history = model.fit(trainDataX[j * step_data_size : (j+1) *  step_data_size], trainDataY[j * step_data_size : (j+1) *  step_data_size ], epochs=1, steps_per_epoch=16, shuffle=True, validation_steps=4, validation_data=(validDataX, validDataY))


    # In[5]:


    model = keras.models.load_model("yolo.h5")


    # In[9]:


    y_pred = model.predict(testDataX)
    roc_auc_score(testDataY.argmax(axis=1), y_pred.argmax(axis=1))

    return     roc_auc_score(testDataY.argmax(axis=1), y_pred.argmax(axis=1))



# # In[13]:


# y_pred = model.predict(validDataX[0:5000])
# roc_auc_score(validDataY[0:5000].argmax(axis=1), y_pred.argmax(axis=1))


# # In[ ]:


# dropout = 0.5


# # In[ ]:


# x = Input(((500, 21, 1)))
# y1 = Conv2D(50, (3,3),  activation="relu",)(x)
# y1 = MaxPool2D(pool_size=(2, 1),)(y1)
# y1 = Dropout(dropout)(y1)
# y1 = Conv2D(200, (3,3), activation="relu",)(y1)
# y1 = MaxPool2D(pool_size=(2, 1),)(y1)
# y1 = Dropout(dropout)(y1)
# y1 = Flatten()(y1)


# # In[ ]:


# y0 = Conv2D(50, (2,2),  activation="relu",)(x)
# y0 = MaxPool2D(pool_size=(2, 1),)(y0)
# y0 = Dropout(dropout)(y0)
# y0 = Conv2D(200, (2,2), activation="relu",)(y0)
# y0 = MaxPool2D(pool_size=(2, 1),)(y0)
# y0 = Dropout(dropout)(y0)
# y0 = Conv2D(200, (2,2), activation="relu",)(y0)
# y0 = MaxPool2D(pool_size=(2, 1),)(y0)
# y0 = Dropout(dropout)(y0)
# y0 = Flatten()(y0)


# # In[ ]:


# y2 = Conv2D(50, (4,4),  activation="relu",)(x)
# y2 = MaxPool2D(pool_size=(2, 1),)(y2)
# y2 = Dropout(dropout)(y2)
# y2 = Conv2D(200, (4,2), activation="relu",)(y2)
# y2 = MaxPool2D(pool_size=(2, 1),)(y2)
# y2 = Dropout(dropout)(y2)
# y2 = Conv2D(200, (4,2), activation="relu",)(y2)
# y2 = MaxPool2D(pool_size=(2, 1),)(y2)
# y2 = Dropout(dropout)(y2)
# y2 = Conv2D(200, (4,2), activation="relu",)(y2)
# y2 = MaxPool2D(pool_size=(2, 1),)(y2)
# y2 = Dropout(dropout)(y2)
# y2 = Flatten()(y2)


# # In[ ]:


# y3 = Conv2D(50, (5,5),  activation="relu",)(x)
# y3 = MaxPool2D(pool_size=(2, 1),)(y3)
# y3 = Dropout(dropout)(y3)
# y3 = Conv2D(200, (5,5), activation="relu",)(y3)
# y3 = MaxPool2D(pool_size=(2, 1),)(y3)
# y3 = Dropout(dropout)(y3)
# y3 = Flatten()(y3)


# # In[ ]:


# y = Concatenate()([y0, y1, y2, y3])
# y = Dense(units=2)(y)
# model2 = Model(inputs=x, outputs =y)


# # In[ ]:


# from keras.optimizers import Adam
# adam = Adam(lr=0.002)
# model2.compile(adam, loss="categorical_crossentropy", metrics=["binary_accuracy"])


# # In[ ]:


# model2.summary()


# # In[ ]:


# from keras.callbacks import EarlyStopping, ModelCheckpoint
# cb = [EarlyStopping(patience=10), ModelCheckpoint("yolo.h5",save_best_only=True)]


# # In[ ]:


# history = model2.fit(trainDataX, trainDataY, epochs=1000, callbacks=cb, validation_data=(validDataX, validDataY), )


# # In[ ]:


# model2 = keras.models.load_model2("yolo.h5")


# # In[ ]:


# y_pred = model2.predict(testDataX)
# roc_auc_score(testDataY.argmax(axis=1), y_pred.argmax(axis=1))


# # In[ ]:
