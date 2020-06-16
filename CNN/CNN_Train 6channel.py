# from __future__ import absolute_import, division, print_function, unicode_literals
##################################################
################ Import Libraries ################
import matplotlib.pyplot as plt
import pickle
import sys
import os
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, concatenate, Dropout, Flatten, Dense
from keras.layers.advanced_activations import ReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import data_from_csv, create_tensor_data, create_model, save_model, TimingCallback, data_from_csv_nonimage, read_data, data_from_csv_nonimage_6channel, create_tensor_data_6channel, create_model_6channel
keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

##################################################
################ Global Variables ################
# Transmitted wave  = 100 cycle Burst, 25ms Burst Period, 40kHz Sine, 10V
id                  = '#1'
DIM                 = '10x10'
RUN_NAME            = 'CNN_stacked_id-' + id + '_' + DIM
ADD_COMMENT         = 'w30_measurements, TEST'
nMEASUREMENTS       = '1,000'
TAG                 = '_6class'
CSV_FILEPATH        = 'data/train_labels' + TAG + '.txt'
DATA_FILEPATH       = 'data/train/complexbaseband/' + DIM + TAG + '/'
BATCH_SIZE          = 100
NUMBER_OF_CLASSES   = 6
EPOCHS              = 5
SAVE_BEST_WEIGHTS   = "CNN/models/weights_" + DIM + "_" + id + ".best.hdf5"
SAVE_MODEL          = "CNN/models/model_" + DIM + "_" + id + ".json"
SAVE_MODEL_HISTORY  = "CNN/models/trainingHistoryDict/trainHistoryDict_" + DIM + "_" + id + "" 
object_classes      = ['noObject','circle','square','rect','Lrect','triangle']
COMMENTS            = 'comments here' + ADD_COMMENT


##################################################
################ Load Data #######################
print('Load Data...')
train_df, input_shape = data_from_csv_nonimage_6channel(CSV_FILEPATH, DATA_FILEPATH)
print('Create Tensor Data...')
x_train, x_test, y_train, y_test = create_tensor_data_6channel(train_df, input_shape, NUMBER_OF_CLASSES)

##################################################
################ Train Model #####################
# import wandb
# from wandb.keras import WandbCallback
# wandb.init(name=RUN_NAME, project="6_class", notes=COMMENTS)

checkpoint = ModelCheckpoint(SAVE_BEST_WEIGHTS, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
cb = TimingCallback()
# callbacks = [WandbCallback(), checkpoint, cb]
callbacks = [checkpoint, cb]
model = create_model_6channel(input_shape, NUMBER_OF_CLASSES)
model.summary()

history = model.fit(x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=True)

# model.save(os.path.join(wandb.run.dir, "model.h5"))

trainingTime = sum(cb.logs)
trainingTimeHours = trainingTime//(60*60)
trainingTime = trainingTime%(60*60)
trainingTimeMinutes =  trainingTime//(60)
trainingTime =  trainingTime%(60)
trainingTimeSeconds = trainingTime
print("Total Training Time: {0:8.0f} hours, {1:8.0f} minutes, {2:8.2f} seconds".format(trainingTimeHours,trainingTimeMinutes,trainingTimeSeconds))

save_model(model, SAVE_MODEL)
with open(SAVE_MODEL_HISTORY, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

fig, axs = plt.subplots(2)
# summarize history for accuracy
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='upper left')
# summarize history for loss
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper left')
plt.show()