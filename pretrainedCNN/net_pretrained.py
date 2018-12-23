import sys
import os
from os.path import join
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten,Input,MaxPooling2D,Dropout,AveragePooling2D,GlobalAveragePooling2D
from keras.callbacks import CSVLogger,TensorBoard,ModelCheckpoint
from keras.backend import flatten
import pickle
sys.path.append(os.path.abspath("utils"))
import dbloader
from load_save import save_model,load_model


model_path = join("raw","pretrCNN.h5")
tensorboardpath = join('tmp','pretrCNN')

class pretrainedCNN:

    def __init__(self,img_shape,num_classes):

        self.trained = True
        self.clf = load_model(model_path)
        if not self.clf:
            print("Creating Model")
            self.trained=False
            base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=img_shape)
            for layer in base_model.layers:
                layer.trainable=False

            bout = base_model.output
            avg_pool = GlobalAveragePooling2D()(bout)
            o = Dense(name='clf_output',units=num_classes,activation='softmax')(avg_pool)

            self.clf = Model(inputs=base_model.input,outputs=o)
            self.clf.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
            self.clf.summary()

    def fit(self,X,Y,Xval,Yval):
        if self.trained:
            print("model already trained")
            return
        self.clf.fit(   X, Y,
                        batch_size=16,
                        validation_data=(Xval,Yval),
                        epochs=100, verbose=1, shuffle=True,
                        callbacks=[ TensorBoard(log_dir=tensorboardpath),
                                    ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

    def predict(self,X):
        y = self.clf.predict(X)
        return y   
