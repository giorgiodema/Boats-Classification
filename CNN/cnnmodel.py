from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Model
from keras.callbacks import CSVLogger
import os
import pickle
from datetime import datetime

def save_classifier(path,clf):
    with open(path,"wb") as f:
        pickle.dump(clf,f)

def load_classifier(path):
    clf = None
    if os.path.exists(path):
        with open(path,"rb") as f:
            clf = pickle.load(f)
    return clf


class ConvolutionalNN:
    def __init__(self,img_shape, num_cat):
        
        # The input layer is a matrix with shape(240,800,3)
        x = Input(shape=img_shape)
        # 6 filters (5,5)   --> shape = (240,800); parameters = 100
        f = Conv2D(filters=6, kernel_size=(5,5),padding='same',activation='relu')(x)
        # MaxPooling (2,2)  --> shape = (120,400,3)
        p = MaxPool2D(pool_size=(2,2))(f)
        # 9 filters (5,5)   --> shape = (120,400,9); parameters = (5*5*8)*4 = 800
        f2 = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(p)
        # MaxPooling (2,2)--> shape = (60,200,9)
        p2 = MaxPool2D(pool_size=(2,2))(f2)
        # 12 filters (5,5)   --> shape = (60,200,12); parameters = (5*5*8)*4 = 800
        f3 = Conv2D(filters=9,kernel_size=(5,5),padding='same',activation='relu')(p2)
        # MaxPooling (5,5)--> shape = (12,40,9)
        p3 = MaxPool2D(pool_size=(5,5))(f3)

        # Flatten  --> output shape = (12*40*9) = 4320
        fl = Flatten()(p3)
        # Dropout
        dr1 = Dropout(rate=0.4,seed=None)(fl)
        # FC layer --> parameters = 4320*864 = 1'440'000
        d1 = Dense(units=864,activation='relu')(dr1)
        #Dropout
        dr2 = Dropout(rate=0.4,seed=None)(d1)
        # FC layer --> parameters = 864*432 = 60'000
        d2 = Dense(units=432,activation='relu')(dr2)
        output = Dense(units=num_cat,activation='softmax')(d2)
        # TOTAL TRAINABLE PARAMETERS --> 1'503'780

        self.model = Model(inputs=x,outputs=output)
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
    def fit(self,X,Y,Xval,Yval):
        self.model.fit(X,Y, batch_size=64, validation_data=(Xval,Yval), epochs=20, verbose=1, shuffle=True, callbacks=[CSVLogger("CNNlogger.csv", separator=',', append=False)])

    def predict(self,X):
        return self.model.predict(X)

    