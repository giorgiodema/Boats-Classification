import sys
import os
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten,Input,MaxPooling2D
from keras.callbacks import CSVLogger
from keras.backend import flatten
import pickle
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


def save_classifier(path,clf):
    with open(path,"wb") as f:
        pickle.dump(clf,f)

def load_classifier(path):
    clf = None
    if os.path.exists(path):
        with open(path,"rb") as f:
            clf = pickle.load(f)
    return clf

class pretrainedCNN:

    def __init__(self,img_shape,num_classes):
        base_model = InceptionV3(weights='imagenet',include_top=True,input_shape=img_shape)
        for layer in base_model.layers:
            layer.trainable=False
        print("ciao ciao")
        base_model.layers.pop()
        bout = base_model.output
        x = Dense(name='clf_dense',units=1024,activation="relu")(bout)
        x = Dense(name='clf_dens2e',units=1024,activation="relu")(x)
        o = Dense(name='clf_output',units=num_classes,activation='softmax')(x)

        self.clf = Model(inputs=base_model.input,outputs=o)
        self.clf.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.clf.summary()

    def fit(self,X,Y,Xval,Yval):
        self.clf.fit(X,Y, batch_size=16, validation_data=(Xval,Yval), epochs=20, verbose=1, shuffle=True, callbacks=TensorBoard(log_dir=tensorboardpath),ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

    def predict(self,X):
        y = self.clf.predict(X)
        return y   
