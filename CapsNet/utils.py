import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pandas as pd
from skimage.io import imread
from keras import backend as K
from scipy import io

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


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

def data_from_csv_nonimage_6channel(csv_filepath, data_filepath):
    train_df = pd.read_csv(csv_filepath, index_col=0)
    train_df['channelbcd'] = train_df.index.map(lambda id: data_filepath + f'/img{id}_bcd.mat')
    C = io.loadmat(train_df.channelbcd.values[0])
    input_dim = C['exportimgs']
    return train_df, input_dim.shape

def plot_log(filename, show=True):

    data = pd.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

if __name__=="__main__":
    plot_log('result/log.csv')