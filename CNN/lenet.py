import sys, os, datetime
import numpy as np
import argparse

import keras
from keras import models, layers, backend
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D

models_dir = 'models'



    
def LeNet(input_shape, num_classes):
    
    model = models.Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model

