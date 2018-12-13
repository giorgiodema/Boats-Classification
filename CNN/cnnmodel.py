from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model
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
        # 64 filters --> shape = (240,800)
        f = Conv2D(filters=64, kernel_size=(24,80),activation='relu')(x)
        # MaxPooling --> shape = (120,400)
        p = MaxPool2D(pool_size=(2,2))(f)
        # 16 filters --> shape = (120,400)
        f2 = Conv2D(filters=16,kernel_size=(6,20),activation='relu')(p)
        # MaxPooling --> shape = (30,100)
        p2 = MaxPool2D(pool_size=(4,4))(f2)
        # Flatten    --> shape = (3000)
        fl = Flatten()(p2)
        d1 = Dense(units=1000,activation='relu')(fl)
        d2 = Dense(units=500,activation='relu')(d1)
        output = Dense(units=num_cat,activation='softmax')(d2)

        self.model = Model(inputs=x,outputs=output)
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    def fit(self,X,Y,Xval,Yval):
        self.model.fit(X,Y, batch_size=64, validation_data=(Xval,Yval), epochs=20, verbose=1, shuffle=True)

    def predict(self,X):
        return self.model.predict(X)

    