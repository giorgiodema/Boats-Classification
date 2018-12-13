import sys
import os
import pickle
import numpy as np
from pdb import set_trace as bp
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


# tmp

img_shape = (240,800,3)
num_classes = 24

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)



Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)


Xtrain_feat = base_model.predict(Xtrain, verbose=1)
Xtrain = Xtrain_feat
del base_model

# end tmp



img_shape = (240,800,3)
num_classes = 24
model_filename = "inceptionV3_pretrained.h5"

model = load_save.load_model(model_filename)
if not model:
    print("Creating model...")

    feat_input = Input(shape=(6,23,2048,))  #shape of last inceptionV3 layer
    x = feat_input

    '''
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(800,240,3))

    # add a global spatial average pooling layer
    x = base_model.output
    bp()
    '''
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=feat_input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    #for layer in base_model.layers:
    #  layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


#Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
#Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)
#manz = model.evaluate(Xtest, Ytest)
#print(manz)
#raise Exception()

# Training set

'''
Xtrain1 = pickle.load(open("features_pretrained_0_2000.pickle", "rb"))
Xtrain2 = pickle.load(open("features_pretrained_2000_end.pickle", "rb"))
Xtrain = np.concatenate((Xtrain1, Xtrain2))
del Xtrain1
del Xtrain2
Ytrain = pickle.load(open("features_pretrained_Y.pickle", "rb"))

# Test set
Xtest = pickle.load(open("features_pretrained_test.pickle", "rb"))
Ytest = pickle.load(open("features_pretrained_test_Y.pickle", "rb"))
'''

#bp()

try:                             #validation_data = (Xtest,Ytest)
    model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
except KeyboardInterrupt:
    # Save the model
    print("Saving...")
    load_save.save_model(model, model_filename)
    quit()


# Save the model
print("Saving...")
load_save.save_model(model, model_filename)
quit()
