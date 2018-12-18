import sys
import os
from os.path import join
import pickle
from keras.preprocessing import image
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.backend import reshape,flatten
from keras.callbacks import CSVLogger, TensorBoard
from keras.optimizers import Adadelta, SGD

import sklearn.svm
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader


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
        input_img = Input(shape=img_shape)  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same',name='encoded')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        #ada = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        #sgd = SGD(lr=0.2, momentum=0.1, decay=0.001, nesterov=False)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])
        self.encoder = Model(input_img,encoded)
        self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
        self.autoencoder.summary()
        self.clf = sklearn.svm.LinearSVC(max_iter=100000)

    def fit(self,X,y,Xval):
        print("Training encoder...")
        tensorboardpath = join('tmp','autoencoder')
        self.autoencoder.fit(X,X,
                        epochs=2,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(Xval, Xval),
                        callbacks=[TensorBoard(log_dir=tensorboardpath), CSVLogger(filename="encoder.csv")])
        x_encoded = self.encoder.predict(X)
        x_encoded = reshape(x_encoded,(self.encoded_shape[0]*self.encoded_shape[1]*self.encoded_shape[2]))
        print("training svm")
        self.clf.fit(x_encoded,y)


    def predict(self,X):

        x_encoded = self.encoder.predict(X)
        x_encoded = reshape(x_encoded,(self.encoded_shape[0]*self.encoded_shape[1]*self.encoded_shape[2],))
        y = self.clf.predict(x_encoded)
        return y
