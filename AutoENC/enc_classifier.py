import sys
import os
from os.path import join
import pickle
from keras.preprocessing import image
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense, Reshape
from keras.models import Model
from keras.backend import reshape,flatten
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from keras.optimizers import Adadelta, SGD

import sklearn.svm
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save

model_path = join("raw","encoder03.h5")


def save_classifier(path,clf):
    with open(path,"wb") as f:
        pickle.dump(clf,f)

def load_classifier(path):
    clf = None
    if os.path.exists(path):
        with open(path,"rb") as f:
            clf = pickle.load(f)
    return clf

class AutoEncSVMclassifier:

    def __init__(self,img_shape, num_cat):

        self.autoencoder = load_save.load_model(model_path)
        if self.autoencoder:
            print("Loading saved clf...")
            self.autoencoder.summary()
            self.encoder = Model(self.autoencoder.get_layer(name='input'),self.autoencoder.get_layer(name='encoded'))
            self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
            self.clf = sklearn.svm.LinearSVC(max_iter=100000)
            return
        print("Creating clf...")
        input_img = Input(shape=img_shape, name='input')  # adapt this if using `channels_first` image data format
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 4), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((10, 10), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(units=3*10*64,activation='relu')(x)
        encoded = Dense(units=960,activation='relu',name='encoded')(x)


        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Dense(units=3*10*64)(encoded)
        x = Reshape((3,10,64))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((10, 10))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu',padding='same')(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
        self.encoder = Model(input_img,encoded)
        self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
        self.autoencoder.summary()
        self.clf = sklearn.svm.LinearSVC(max_iter=100000,verbose=1)

    def fit(self,X,y,Xval):
        print("Training encoder...")
        tensorboardpath = join('tmp','autoencoder')
        try:
            self.autoencoder.fit(X,X,
                            epochs=100,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(Xval, Xval),
                            callbacks=[TensorBoard(log_dir=tensorboardpath), CSVLogger(filename="encoder.csv"),ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
        except KeyboardInterrupt:
            if not os.path.exists(model_path):
                load_save.save_model(self.autoencoder,model_path)
        x_encoded = self.encoder.predict(X)
        print("training svm")
        self.clf.fit(x_encoded,y)


    def predict(self,X):

        x_encoded = self.encoder.predict(X)
        y = self.clf.predict(x_encoded)
        return y
