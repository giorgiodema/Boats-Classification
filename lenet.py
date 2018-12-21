import keras
import sys, os, datetime
import numpy as np
import keras
from keras import models, layers, backend, optimizers
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save
import alexnet
import lenet

img_shape = (240,800,3)
num_classes = 23
model_filename = os.path.join("trained_models","LeNet.h5")


def LeNet(input_shape, num_classes):
    
    model = models.Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(5, 5), padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = optimizers.Adam(lr=0.0001) #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model



model = load_save.load_model(model_filename)
if not model:
    model = lenet.LeNet(img_shape,num_classes)

model.summary()


Xtrain,Ytrain,img_shape,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

#model.evaluate(Xtest,Ytest)
#quit()

try:
    #model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1, validation_data = (Xtest,Ytest), shuffle=True)
    history = model.fit(Xtrain, Ytrain, batch_size=32, epochs=30, verbose=1, validation_data=(Xtest,Ytest), shuffle="batch")
except KeyboardInterrupt:
    # Save the model
    print("Saving...")
    load_save.save_model(model, model_filename)
    quit()
print("Training Completed...")
print("Saving...")
load_save.save_model(model, model_filename)
print("Done...")
    


