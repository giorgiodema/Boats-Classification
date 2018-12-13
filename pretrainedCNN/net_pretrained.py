import sys
import os
import pickle
import numpy as np
from random import shuffle
from pdb import set_trace as bp
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (240,800,3)
num_classes = 24
model_filename = "inceptionV3_pretrained.h5"


model = load_save.load_model(model_filename)
if not model:
    print("Creating model...")

    feat_input = Input(shape=(6,23,2048,))  #shape of last inceptionV3 layer
    x = feat_input

    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=feat_input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


#Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
#Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)
#manz = model.evaluate(Xtest, Ytest)
#print(manz)
#raise Exception()

# Training set


Xtrain1 = pickle.load(open("features_pretrained_0_2000.pickle", "rb"))
Xtrain2 = pickle.load(open("features_pretrained_2000_end.pickle", "rb"))
Xtrain = np.concatenate((Xtrain1, Xtrain2))
del Xtrain1
del Xtrain2
Ytrain = pickle.load(open("features_pretrained_Y.pickle", "rb"))

'''
# Test set
Xtest = pickle.load(open("features_pretrained_test.pickle", "rb"))
Ytest = pickle.load(open("features_pretrained_test_Y.pickle", "rb"))
'''

#bp()

# shuffle
aux = [(x,y) for x,y in zip(Xtrain, Ytrain)]
shuffle(aux)
Xtrain = np.array([x[0] for x in aux])
Ytrain = np.array([x[1] for x in aux])

try:                             #validation_data = (Xtest,Ytest)
    model.fit(Xtrain, Ytrain, batch_size=16, epochs=10, verbose=1, validation_split=0.2, shuffle=False)
except KeyboardInterrupt:
    # Save the model
    print("Saving...")
    load_save.save_model(model, model_filename)
    quit()


# Save the model
print("Saving...")
load_save.save_model(model, model_filename)
quit()
