import sys
import os
from os.path import join
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense, Reshape,Dropout
from keras.models import Model
from keras.backend import reshape,flatten
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from keras.optimizers import Adadelta, SGD

import sklearn.svm
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save

model_path = join("raw","encoder.h5")
tensorboardpath = join('tmp','autoencoder')

class AutoEncSVMclassifier:

    def __init__(self,img_shape, num_cat):

        self.autoencoder = load_save.load_model(model_path)
        self.trained = True
        if self.autoencoder:
            print("Loading saved clf...")
            self.autoencoder.summary()
            self.encoder = Model(self.autoencoder.get_layer(name='input'),self.autoencoder.get_layer(name='encoded'))
            self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
            self.clf = sklearn.svm.LinearSVC(max_iter=100000)
            return

        self.trained = False
        print("Creating clf...")

        # The input layer is a matrix with shape(240,800,3)
        input_img = Input(shape=img_shape, name='input')
        # 21 filters (5,5)   --> shape = (240,800)
        x = Conv2D(filters=21, kernel_size=(5,5),padding='same',activation='relu')(input_img)
        # MaxPooling (2,2)  -->  shape = (120,400,21)
        x = MaxPooling2D(pool_size=(2,2))(x)
        # 18 filters (5,5)   -->  shape = (120,400,18)
        x = Conv2D(filters=18,kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (2,2)-->    shape = (60,200,18)
        x = MaxPooling2D(pool_size=(2,2))(x)
        # 9 filters (5,5)   --> shape = (60,200,9)
        x = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (4,4)--> shape = (15,50,9)
        x = MaxPooling2D(pool_size=(4,4))(x)

        # Flatten  --> output shape = (15*50*9) = 6750
        encoded = Flatten(name='encoded')(x)

        # shape = (15,50,9)
        x = Reshape((15,50,9))(encoded)
        # shape = (15,50,9)
        x = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (60,200,9)
        x = UpSampling2D(size=(4,4))(x)
        # shape = (60,200,15)
        x = Conv2D(filters=15,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (120,400,15)
        x = UpSampling2D(size=(2,2))(x)
        # shape = (120,400,21)
        x = Conv2D(filters=21,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (240,800,21)
        x = UpSampling2D(size=(2,2))(x)
        # shape = (240,480,3)
        decoded = Conv2D(filters=3,kernel_size=(5,5),padding='same',activation='sigmoid')(x)



        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        self.encoder = Model(input_img,encoded)
        self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
        self.autoencoder.summary()
        self.clf = sklearn.svm.LinearSVC(max_iter=100000,verbose=1)

    def fit(self,X,y,Xval):
        
        if not self.trained:
            print("training encoder")
            self.autoencoder.fit(X,X,
                            epochs=40,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(Xval, Xval),
                            callbacks=[ TensorBoard(log_dir=tensorboardpath),
                                        ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

        x_encoded = self.encoder.predict(X)
        print("training svm")
        self.clf.fit(x_encoded,y)


    def predict(self,X):

        x_encoded = self.encoder.predict(X)
        y = self.clf.predict(x_encoded)
        return y
