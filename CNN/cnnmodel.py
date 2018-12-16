from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model
from keras.callbacks import CSVLogger
import os
import pickle

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
        d1 = Dense(units=300,activation='relu')(fl)
        # FC layer --> parameters = 300*200 = 60'000
        d2 = Dense(units=200,activation='relu')(d1)
        output = Dense(units=num_cat,activation='softmax')(d2)
        # TOTAL TRAINABLE PARAMETERS --> 1'503'780

        self.model = Model(inputs=x,outputs=output)
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    def fit(self,X,Y,Xval,Yval):
        self.model.fit(X,Y, batch_size=64, validation_data=(Xval,Yval), epochs=20, verbose=1, shuffle=True, callbacks=[CSVLogger("CNNlogger.csv", separator=',', append=False)])

    def predict(self,X):
        return self.model.predict(X)

    