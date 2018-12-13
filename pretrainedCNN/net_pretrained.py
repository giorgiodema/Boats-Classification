import sys
import os
import pickle
import numpy as np
import random
from pdb import set_trace as bp
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


random.seed(42)


img_shape = (240,800,3)
num_classes = 24
model_filename = "inceptionV3_pretrained.h5"


model = load_save.load_model(model_filename)
if not model:
    print("Creating model...")

    feat_input = Input(shape=(2048,))  #shape of last inceptionV3 layer
    x = feat_input

    #x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=feat_input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()



def shuffle_input(X,Y):
    aux = [(x,y) for x,y in zip(X, Y)]
    random.shuffle(aux)
    X_shffld = np.array([x[0] for x in aux])
    Y_shffld = np.array([x[1] for x in aux])
    return X_shffld, Y_shffld


def load_train_features():
    Xtrain1 = pickle.load(open("features_pretrained_0_2000.pickle", "rb"))
    Xtrain2 = pickle.load(open("features_pretrained_2000_end.pickle", "rb"))
    Xtrain = np.concatenate((Xtrain1, Xtrain2))
    del Xtrain1
    del Xtrain2
    Ytrain = pickle.load(open("features_pretrained_Y.pickle", "rb"))
    return shuffle_input(Xtrain, Ytrain)

def load_test_features():
    Xtest = pickle.load(open("features_pretrained_test.pickle", "rb"))
    Ytest = pickle.load(open("features_pretrained_test_Y.pickle", "rb"))
    return shuffle_input(Xtest, Ytest)




Xtrain, Ytrain = load_train_features()


                            #validation_data = (Xtest,Ytest)
model.fit(Xtrain, Ytrain, batch_size=16, epochs=10, verbose=1, validation_split=0.2, shuffle=False)


# Save the model
print("Saving...")
load_save.save_model(model, model_filename)
quit()
