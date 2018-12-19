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

model_path = join("raw","encoder04.h5")


def save_classifier(path,clf):
    with open(path,"wb") as f:
        joblib.dump(clf,f)

def load_classifier(path):
    clf = None
    if os.path.exists(path):
        with open(path,"rb") as f:
            clf = joblib.load(f)
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

        # The input layer is a matrix with shape(240,800,3)
        input_img = Input(shape=img_shape, name='input')
        # 6 filters (5,5)   --> shape = (240,800); parameters = 100
        x = Conv2D(filters=6, kernel_size=(5,5),padding='same',activation='relu')(input_img)
        # MaxPooling (2,2)  --> shape = (120,400,3)
        x = MaxPooling2D(pool_size=(2,2))(x)
        # 9 filters (5,5)   --> shape = (120,400,9); parameters = (5*5*8)*4 = 800
        x = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (2,2)--> shape = (60,200,9)
        x = MaxPooling2D(pool_size=(2,2))(x)
        # 12 filters (5,5)   --> shape = (60,200,12); parameters = (5*5*8)*4 = 800
        x = Conv2D(filters=12,kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (5,5)--> shape = (12,40,12)
        x = MaxPooling2D(pool_size=(5,5))(x)

        # Flatten  --> output shape = (12*40*12) = 5760
        x = Flatten()(x)
        # Dropout
        x = Dropout(rate=0.4,seed=None)(x)
        # FC layer --> parameters = 4320*2160 = 1'440'000
        encoded = Dense(units=2880,activation='relu',name='encoded')(x)
 
        x = Dropout(rate=0.4,seed=None)(encoded)
        # shape = 4320
        x = Dense(units=5760,activation='relu')(x)
        # shape = (12,40,9)
        x = Reshape((12,40,12))(x)
        # shape = (12,40,9)
        x = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (60,200,9)
        x = UpSampling2D(size=(5,5))(x)
        # shape = (60,200,9)
        x = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (120,400,9)
        x = UpSampling2D(size=(2,2))(x)
        # shape = (120,400,3)
        x = Conv2D(filters=3,kernel_size=(5,5),padding='same',activation='relu')(x)
        # shape = (240,800,3)
        decoded = UpSampling2D(size=(2,2))(x)




        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
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
                            batch_size=8,
                            shuffle=True,
                            validation_data=(Xval, Xval),
                            callbacks=[TensorBoard(log_dir=tensorboardpath), CSVLogger(filename="encoder.csv"),ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
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
