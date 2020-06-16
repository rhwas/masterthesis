from skimage.io import imread
from scipy import io
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
import numpy as np
import pandas as pd
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, concatenate, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import ReLU, LeakyReLU
from timeit import default_timer as timer
keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def data_from_csv(csv_filepath, data_filepath):
    train_df = pd.read_csv(csv_filepath, index_col=0)
    train_df['channelb'] = train_df.index.map(lambda id: data_filepath + f'img{id}_b.png')
    train_df['channelc'] = train_df.index.map(lambda id: data_filepath + f'img{id}_c.png')
    train_df['channeld'] = train_df.index.map(lambda id: data_filepath + f'img{id}_d.png')
    input_dim = imread(train_df.channelb.values[0])
    if len(input_dim.shape) == 2:
        input_dim = np.expand_dims(input_dim, axis=2)
    input_shape = input_dim.shape
    return train_df, input_shape

def data_from_csv_nonimage(csv_filepath, data_filepath):
    train_df = pd.read_csv(csv_filepath, index_col=0)
    train_df['channelb'] = train_df.index.map(lambda id: data_filepath + f'/img{id}_b.mat')
    train_df['channelc'] = train_df.index.map(lambda id: data_filepath + f'/img{id}_c.mat')
    train_df['channeld'] = train_df.index.map(lambda id: data_filepath + f'/img{id}_d.mat')
    C = io.loadmat(train_df.channelb.values[0])
    input_dim = C['exportimgs']
    input_dim = input_dim[:,:,:2]
    if len(input_dim.shape) == 2:
        input_dim = np.expand_dims(input_dim, axis=2)
    input_shape = input_dim.shape
    print(input_shape)
    return train_df, input_shape

def read_images(file_paths, input_shape):
    img_rows, img_cols, channels = input_shape
    images = []
    for file_path in file_paths:
        images.append(imread(file_path))
    images = np.asarray(images, dtype=np.float32)
    # normalize
    images = images/255.0 #np.max(images)
    # reshape to match Keras expectaions
    # print(len(images))
    # print(images.shape)
    images = images.reshape(images.shape[0], img_rows, img_cols, channels)
    return images

def data_from_csv_nonimage_6channel(csv_filepath, data_filepath):
    train_df = pd.read_csv(csv_filepath, index_col=0)
    train_df['channelbcd'] = train_df.index.map(lambda id: data_filepath + f'/img{id}_bcd.mat')
    C = io.loadmat(train_df.channelbcd.values[0])
    input_dim = C['exportimgs']
    return train_df, input_dim.shape

def read_data(file_paths, input_shape):
    rows, cols, channels = input_shape
    images = []
    for file_path in file_paths:
        C = io.loadmat(file_path)
        img = C['exportimgs']
        images.append(img[:,:,:channels])
    images = np.asarray(images, dtype=np.float32)
    orig_shape = images.shape
    print(orig_shape)
    
    if images.max() != 1.0:
        # normalize
        images = images.reshape(images.shape[0]*rows*cols*channels)
        images = images.reshape(-1,1)
        print('Before norm max: {}'.format(images.max()))
        images = images/images.max()
        print('After norm max: {}'.format(images.max()))
        images = images.reshape(orig_shape)
        images = images.reshape(images.shape[0], rows, cols, channels)

    return images

def create_tensor_data_images(train_df, input_shape, NUMBER_OF_CLASSES):
    print('...ChannelB...')
    x_train_channelb = read_images(train_df.channelb.values, input_shape)
    print('...ChannelC...')
    x_train_channelc = read_images(train_df.channelc.values, input_shape)
    print('...ChannelD')
    x_train_channeld = read_images(train_df.channeld.values, input_shape)
    # labels - convert class vectors to binary class matrices One Hot Encoding
    labels = train_df.object.values
    labels = keras.utils.to_categorical(labels, NUMBER_OF_CLASSES)
    x_train_comp = np.stack((x_train_channelb, x_train_channelc, x_train_channeld), axis=4)
    x_train, x_test, y_train, y_test = train_test_split(x_train_comp, labels, test_size = 0.3, random_state=None)
    return x_train, x_test, y_train, y_test

def create_tensor_data(train_df, input_shape, NUMBER_OF_CLASSES):
    print('...ChannelB...')
    x_train_channelb = read_data(train_df.channelb.values, input_shape)
    print('...ChannelC...')
    x_train_channelc = read_data(train_df.channelc.values, input_shape)
    print('...ChannelD')
    x_train_channeld = read_data(train_df.channeld.values, input_shape)
    # labels - convert class vectors to binary class matrices One Hot Encoding
    labels = train_df.object.values
    labels = keras.utils.to_categorical(labels, NUMBER_OF_CLASSES)
    x_train_comp = np.stack((x_train_channelb, x_train_channelc, x_train_channeld), axis=4)
    x_train, x_test, y_train, y_test = train_test_split(x_train_comp, labels, test_size = 0.3, random_state=None)
    return x_train, x_test, y_train, y_test

def create_tensor_data_6channel(train_df, input_shape, NUMBER_OF_CLASSES):
    print('...ChannelBCD')
    x_train_channelbcd = read_data(train_df.channelbcd.values, input_shape)
    # labels - convert class vectors to binary class matrices One Hot Encoding
    labels = train_df.object.values
    labels = keras.utils.to_categorical(labels, NUMBER_OF_CLASSES)
    x_train, x_test, y_train, y_test = train_test_split(x_train_channelbcd, labels, test_size = 0.3, random_state=None)
    return x_train, x_test, y_train, y_test

##################################################
################ Model ###########################

def create_convolution_layers(input_img, input_shape):
    model = Conv2D(64, (3, 3), padding='same', input_shape=input_shape, use_bias=False)(input_img)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Dropout(0.2)(model)

    model = Conv2D(64, (3, 3), padding='same', input_shape=input_shape, use_bias=False)(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Dropout(0.2)(model)

    model = MaxPooling2D((2, 2),padding='same')(model)

    model = Conv2D(128, (5, 5), padding='same', use_bias=True)(model)
    model = ReLU()(model)
    model = Dropout(0.2)(model)

    # model = Conv2D(128, (2, 2), padding='same', input_shape=input_shape, use_bias=True)(input_img)
    # model = ReLU()(model)
    # model = Dropout(0.2)(model)
    return model
    
def create_model(input_shape, NUMBER_OF_CLASSES):
    channelb_input = Input(shape=input_shape)
    channelc_input = Input(shape=input_shape)
    channeld_input = Input(shape=input_shape)
    channelb_model = create_convolution_layers(channelb_input, input_shape)
    channelc_model = create_convolution_layers(channelc_input, input_shape)
    channeld_model = create_convolution_layers(channeld_input, input_shape)

    conv = concatenate([channelb_model, channelc_model, channeld_model])
    conv = Flatten()(conv)
    dense = Dense(32, use_bias=False)(conv)
    dense = BatchNormalization()(dense)
    dense = ReLU()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(64, use_bias=True)(dense)
    dense = ReLU()(dense)
    
    
    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(dense)
    model = Model(inputs=[channelb_input, channelc_input, channeld_input], outputs=[output])
    # model = Model(inputs=[channelbcd_input], outputs=[output])

    # opt = optimizers.Adam(learning_rate=0.01)
    # opt = optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def create_model_6channel(input_shape, NUMBER_OF_CLASSES):
    channelbcd_input = Input(shape=input_shape)
    channelbcd_model = create_convolution_layers(channelbcd_input, input_shape)

    conv = Flatten()(channelbcd_model)
    dense = Dense(32, use_bias=False)(conv)
    dense = BatchNormalization()(dense)
    dense = ReLU()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(64, use_bias=True)(dense)
    dense = ReLU()(dense)
    
    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(dense)
    model = Model(inputs=[channelbcd_input], outputs=[output])

    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def save_model(model, filepath):
    model_json = model.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    model.save_weights('CNN/models/weights_end.best.hdf5')