# -*- coding: utf-8 -*-
"""cnn_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18slOFhhVV42d-fglwtGG_1AzCKv4YMbr
"""

from google.colab import drive
drive.mount('/content/driver/', force_remount=True)

# !git clone https://github.com/Trongha/SleepApneaWithReccurentPlot.git /content/driver/My\ Drive/SleepApneaWithCrp2

# %cd /content/driver/My\ Drive/SleepApneaWithCrp2/
# !git checkout makeRp

# !pip install cython
# !pip install wfdb
# # !pip install hrv
# !pip install tensorflow
# !pip install keras

from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import sklearn

batch_size = 32
# số lượng ảnh 1 lần lấy đưa vào model
num_classes = 2
# so luong nhan
epochs = 3
# input image dimensions
img_x, img_y = 400, 400
input_shape = (img_x, img_y, 1)

import sys
sys.path.append('/content/driver/My Drive/SleepApneaWithCrp2')
sys.path.append('/content/driver/My Drive/SleepApneaWithCrp2/src')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/driver/My\ Drive/SleepApneaWithCrp2/src

# !git status
# !git log
# !git pull

import MyUtil as myUtil
import numpy as np
import config as config
from tqdm import tqdm
import RecurrentPlot as rp
import RecurrenceQuantificationAnalysis as rqa
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# build model with 1 layer, AVERAGE pooling 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                activation='relu',
                input_shape=input_shape))

model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# build model with 1 layer, MAX pooling (Kết quả tốt nhất)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                activation='relu',
                input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# build model with 2 layer, AVERAGE pooling 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                activation='relu',
                input_shape=input_shape))

model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# build model with 2 layers, MAX POOLING
  model = Sequential()

  model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                  activation='relu',
                  input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
 
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(35, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))

  # compile model
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

# ghi log của mô hình
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []


    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

#Chia record
allRecordNames = config.NAME_OF_RECORD
recordNameForTest = ['a03', 'a05', 'a13', 'a16', 'a19']
recordNameForTrain = [recordName for recordName in allRecordNames if recordName not in recordNameForTest]
print('record for train: ', recordNameForTrain)
print('record for test: ', recordNameForTest)

# train model 1
# load data train

recordNames = config.NAME_OF_RECORD

numTrain = 30
print('load train . . .')
trainRp = []
trainLabel = []

# loadRpByCluster(numberOfCluster, indexCluster, listRecordNames=None, type='train'):
numCluster = 10
for iCluster in range(0, numCluster):
    print(' train cluster ', iCluster, ' . . . ')
    trainRp, trainLabel, _ = myUtil.loadRpByCluster(numCluster, iCluster, recordNameForTrain, 'train')
    # chuẩn hóa data
    trainRp = trainRp.reshape(trainRp.shape[0], img_x, img_y, 1)

    # convert the data to the right type
    x_train = trainRp.astype('float32')
    

    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(trainLabel, num_classes)
    
    # print(trainLabel.shape)

    
    # train model
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[history],
          validation_split=0.3)

testRp = []
testLabel = []
for iRecord, recordName in enumerate(recordNameForTest):
    # load data test
    testRp, testLabel, _ = myUtil.readRpBinary(recordName, 'test')
    
    # chuẩn hóa data
    testRp = testRp.reshape(testRp.shape[0], img_x, img_y, 1)    
    # convert the data to the right type
    x_test = testRp.astype('float32')
    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_test = keras.utils.to_categorical(testLabel, num_classes)


    # predict
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n','Record ', recordName, ':')
    # print('Test loss:', score[0])
    print('Accuracy: %.2f%%' %(score[1]*100))

    
    Y_pred = model.predict(x_test)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # precision and recall
    print('Precision: %.2f%%' %(precision_score(testLabel, y_pred)*100))
    print('Recall: %.2f%%' %(recall_score(testLabel, y_pred)*100))
    # confusion matrix
    print('\n', confusion_matrix(testLabel, y_pred))