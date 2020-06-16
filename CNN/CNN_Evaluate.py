# from __future__ import absolute_import, division, print_function, unicode_literals
##################################################
################ Import Libraries ################
import keras
from keras import optimizers
from keras.models import model_from_json
from utils import data_from_csv, read_images, data_from_csv_nonimage, read_data, data_from_csv_nonimage_6channel
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

id                  = '#1'
TAG                 = '_6class_T2'
DIM                 = '10x10'
CONSTRUCTION_METHOD = 'stacked' # 'image' or 'matrix' or 'stacked'
CSV_FILEPATH        = 'data/test_labels' + TAG + '.txt'
DATA_FILEPATH       = 'data/test/complexbaseband/' + DIM + TAG + '/'
NUMBER_OF_CLASSES   = 6
WEIGHTS             = 'CNN/models/weights_' + DIM + '_' + id + '.best.hdf5'
MODEL               = 'CNN/models/model_' + DIM + '_' + id + '.json'
object_classes      = ['noObject','circle','square','rect','Lrect','triangle']

train_df, input_shape = data_from_csv_nonimage(CSV_FILEPATH, DATA_FILEPATH)

if CONSTRUCTION_METHOD == 'image':
    train_df, input_shape = data_from_csv(CSV_FILEPATH, DATA_FILEPATH)
    x_test_channelb = read_images(train_df.channelb.values, input_shape)
    x_test_channelc = read_images(train_df.channelc.values, input_shape)
    x_test_channeld = read_images(train_df.channeld.values, input_shape)
elif CONSTRUCTION_METHOD == 'matrix':
    train_df, input_shape = data_from_csv_nonimage(CSV_FILEPATH, DATA_FILEPATH)
    x_test_channelb = read_data(train_df.channelb.values, input_shape)
    x_test_channelc = read_data(train_df.channelc.values, input_shape)
    x_test_channeld = read_data(train_df.channeld.values, input_shape)
else:
    train_df, input_shape = data_from_csv_nonimage_6channel(CSV_FILEPATH, DATA_FILEPATH)
    x_test_channelbcd = read_data(train_df.channelbcd.values, input_shape)

labels = train_df.object.values
labels = keras.utils.to_categorical(labels, NUMBER_OF_CLASSES)

# load model
json_file = open(MODEL, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights
model.load_weights(WEIGHTS)
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if CONSTRUCTION_METHOD == 'stacked':
    final_loss, final_acc = model.evaluate(x_test_channelbcd, labels, verbose=1)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
    predictions = model.predict(x_test_channelbcd)
else:
    final_loss, final_acc = model.evaluate([x_test_channelb, x_test_channelc, x_test_channeld], labels, verbose=1)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
    predictions = model.predict([x_test_channelb, x_test_channelc, x_test_channeld])

print(type(predictions))
y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(labels, axis=1)


print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=object_classes))