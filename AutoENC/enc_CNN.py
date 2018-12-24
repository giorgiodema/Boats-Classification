import sys
import os
from os.path import join
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense, Reshape,Dropout,Lambda
from keras.models import Model
from keras.backend import reshape,flatten
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from keras.optimizers import Adadelta, SGD


import sklearn.svm
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save

enc_path = join("raw","encoder.h5")
clf_path = join("raw","enc_clf.h5")
tensorboardpath = join('tmp','autoencoder')


def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        from keras import backend as K
        import tensorflow as tf
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)

class AutoEncClassifier:

    def __init__(self,img_shape, num_cat):
        self.autoencoder = load_save.load_model(enc_path)
        self.clf = load_save.load_model(clf_path)
        self.enc_trained = True
        self.clf_trained = True

        if not self.autoencoder:
            self.enc_trained = False
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
            x = UpSampling2DBilinear(4)(x)
            # shape = (60,200,15)
            x = Conv2D(filters=15,kernel_size=(5,5),padding='same',activation='relu')(x)
            # shape = (120,400,15)
            x = UpSampling2DBilinear(2)(x)
            # shape = (120,400,21)
            x = Conv2D(filters=21,kernel_size=(5,5),padding='same',activation='relu')(x)
            # shape = (240,800,21)
            x = UpSampling2DBilinear(2)(x)
            # shape = (240,480,3)
            decoded = Conv2D(filters=3,kernel_size=(5,5),padding='same',activation='sigmoid')(x)



            self.autoencoder = Model(input_img, decoded)
            self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
            self.encoder = Model(input_img,encoded)
            self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape
            self.autoencoder.summary()
        
        else:
            print("Loading saved enc...")
            self.autoencoder.summary()
            self.encoder = Model(inputs=self.autoencoder.get_layer(name='input').output,outputs=self.autoencoder.get_layer(name='encoded').output)
            self.encoded_shape = self.autoencoder.get_layer(name='encoded').output_shape

        if not self.clf:
            print("Creating clf...")
            self.clf_trained = False
            inp = Input(shape=(6750,))
            x = Dense(units=1000,activation='relu')(inp)
            x = Dropout(rate=0.4)(x)
            x = Dense(units=1000,activation='relu')(x)
            x = Dense(units=500,activation='relu')(x)
            pred = Dense(units=num_cat,activation='softmax')(x)
            self.clf = Model(inputs=inp,outputs=pred)
            self.clf.summary()
            self.clf.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        else:
            print("Loading saved clf...")
            self.clf.summary()

    def fit(self,X,Y,Xval,Yval):
        
        if not self.enc_trained:
            print("training encoder")
            #os.environ["CUDA_VISIBLE_DEVICES"]="0"
            self.autoencoder.fit(X,X,
                            epochs=40,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(Xval, Xval),
                            callbacks=[ TensorBoard(log_dir=tensorboardpath),
                                        ModelCheckpoint(enc_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

        print("training clf")
        x_encoded = self.encoder.predict(X)
        xtest_encoded = self.encoder.predict(Xval)
        if not self.clf_trained:
            self.clf.fit(x_encoded,Y,
                        epochs=40,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(xtest_encoded, Yval),
                        callbacks=[ TensorBoard(log_dir=tensorboardpath),
                                    ModelCheckpoint(clf_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
                        


    def predict(self,X):

        x_encoded = self.encoder.predict(X)
        y = self.clf.predict(x_encoded)
        return y