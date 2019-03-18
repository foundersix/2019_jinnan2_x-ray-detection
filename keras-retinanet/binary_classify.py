#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-03-07 16:13:37
@LastEditTime: 2019-03-17 12:21:44
'''

import copy
import os
import random

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from sklearn.metrics import (f1_score, hamming_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

restrict_img_path = 'keras_retinanet/CSV/data/jinnan2_round1_train_20190305/restricted/'
normal_img_path = 'keras_retinanet/CSV/data/jinnan2_round1_train_20190305/normal/'


def get_filename(path, filetype):
    final_name = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if filetype in file:
                final_name.append(file)
    return final_name


def get_train_file():
    # 加入训练的正常图片与有违禁品图片的比例
    normal_rate = 0.5

    filetype = '.jpg'
    filenames_innormal = get_filename(restrict_img_path, filetype)
    random.shuffle(filenames_innormal)
    print(len(filenames_innormal))

    filenames_normal = get_filename(normal_img_path, filetype)
    random.shuffle(filenames_normal)

    filenames_normal = filenames_normal[:int(
        normal_rate*len(filenames_innormal))]
    print(len(filenames_normal))
    return filenames_innormal, filenames_normal


def img_process(filenames_innormal, filenames_normal):
    input_shape = (256, 256)
    x = []
    y = []

    for img_file in filenames_innormal:
        img = cv2.imread(os.path.join(restrict_img_path, img_file))
        img = cv2.resize(img, input_shape)
        x.append(img)
        y.append(np.array([1]))

    for img_file in filenames_normal:
        img = cv2.imread(os.path.join(normal_img_path, img_file))
        img = cv2.resize(img, input_shape)
        x.append(img)
        y.append(np.array([0]))

    x = np.array(x)
    print(x.shape)
    y = np.array(y)
    print(y.shape)
    y = y.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=100)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def model_train(x_train, y_train, x_test, y_test):
    base_model = keras.applications.vgg19.VGG19(
        weights='imagenet', include_top=False, pooling='avg')

    predictions = Dense(1, activation='sigmoid')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5",
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto')

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 32

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        validation_data=(x_test, y_test),
                        epochs=50,
                        verbose=1,
                        workers=1,
                        callbacks=[check]
                        )


if __name__ == "__main__":

    filenames_innormal, filenames_normal = get_train_file()
    x_train, y_train, x_test, y_test = img_process(
        filenames_innormal, filenames_normal)
    model_train(x_train, y_train, x_test, y_test)
