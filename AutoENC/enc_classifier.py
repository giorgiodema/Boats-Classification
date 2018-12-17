import sys
import os
import pickle
from keras.preprocessing import image
from keras.layers import Input,Dense,UpSampling3D, Conv2D, Conv2DTranspose,MaxPool2D,Flatten,UpSampling2D,Reshape
from keras.models import Model
from keras.backend import reshape,flatten
from keras.callbacks import CSVLogger

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
        print("Creating model...")
        
        # The input layer is a matrix with shape(240,800,3)
        x = Input(shape=img_shape)
        # 4 filters (5,5)   --> shape = (240,800); parameters = 100
        f = Conv2D(filters=4, kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (8,8)  --> shape = (30,100,3)
        p = MaxPool2D(pool_size=(8,8))(f)
        # 8 filters (5,5)   --> shape = (30,100,3); parameters = (5*5*8)*4 = 800
        f2 = Conv2D(filters=8,kernel_size=(5,5),padding='same',activation='relu')(p)
        # MaxPooling (10,10)--> shape = (3,10,3)
        p2 = MaxPool2D(pool_size=(10,10))(f2)
        # Flatten  --> output shape = (3*10*3)*8*4 = 2880
        fl = Flatten()(p2)
        # FC layer --> parameters = 2880*500 = 1'440'000
        d1 = Dense(units=240,activation='relu')(fl)
        # FC layer --> parameters = 300*200 = 60'000
        output = Dense(units=200,activation='relu')(d1)
        invd1 = Dense(units=240,activation='relu')(output)
        invfl = Reshape((3,10,8))(invd1)
        invp2 = UpSampling2D(size=(10,10))(invfl)
        invf2 = Conv2DTranspose(filters=8,kernel_size=(5,5),padding='same',activation='relu')(invp2)
        invp = UpSampling2D((8,8))(invf2)
        invf = Conv2DTranspose(filters=4, kernel_size=(5,5),padding='same',activation='relu')(invp)
        invinp = Conv2DTranspose(filters=3, kernel_size=(5,5),padding='same',activation='relu')(invf)

        self.model = Model(inputs=x,outputs=invinp)
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        self.enc_model = Model(inputs=x,outputs=output)
        self.clf = sklearn.svm.LinearSVC(max_iter=100000)
        
        """
        # The input layer is a matrix with shape(240,800,3)
        x = Input(shape=img_shape)
        # 4 filters (5,5)   --> shape = (240,800); parameters = 100
        f = Conv2D(filters=4, kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (8,8)  --> shape = (30,100,3)
        p = MaxPool2D(pool_size=(8,8))(f)
        # 8 filters (5,5)   --> shape = (30,100,3); parameters = (5*5*8)*4 = 800
        f2 = Conv2D(filters=8,kernel_size=(5,5),padding='same',activation='relu')(p)
        # MaxPooling (10,10)--> shape = (3,10,3)
        p2 = MaxPool2D(pool_size=(10,10))(f2)
        # Flatten  --> output shape = (3*10*3)*8*4 = 2880
        invp2 = UpSampling2D(size=(10,10))(p2)
        invf2 = Conv2DTranspose(filters=8,kernel_size=(5,5),padding='same',activation='relu')(invp2)
        invp = UpSampling2D((8,8))(invf2)
        invf = Conv2DTranspose(filters=4, kernel_size=(5,5),padding='same',activation='relu')(invp)
        invinp = Conv2DTranspose(filters=3, kernel_size=(5,5),padding='same',activation='relu')(invf)

        self.model = Model(inputs=x,outputs=invinp)
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        self.enc_model = Model(inputs=x,outputs=p)
        self.clf = sklearn.svm.LinearSVC(max_iter=100000)
        """


    def fit(self,X,y):
        print("Training encoder...")

        self.model.fit(X,X, batch_size=8, epochs=8, verbose=1, shuffle=True, callbacks=[CSVLogger("ENCMODELlogger.csv", separator=',', append=False)])
        x_encoded = self.enc_model.predict(X)
        print("training svm")
        self.clf.fit(x_encoded,y)


    def predict(self,X):

        x_encoded = self.enc_model.predict(X)
        y = self.clf.predict(x_encoded)
        return y