import sys
import os
import pickle
import numpy as np
import random
from pdb import set_trace
from matplotlib import pyplot as plt
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
num_classes = 23
model_filename = os.path.join("pretrainedCNN","inceptionV3_pretrained.h5")


model = load_save.load_model(model_filename)
if not model:
    print("Creating model...")

    # New model using as input the output of the inception
    feat_input = Input(shape=(2048,))  #shape of last inceptionV3 layer
    x = feat_input

    # Add a fully connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    
    # Add a final layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=feat_input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()




# Performs manual shuffling of data
def shuffle_input(X,Y):
    aux = [(x,y) for x,y in zip(X, Y)]
    random.shuffle(aux)
    X_shffld = np.array([x[0] for x in aux])
    Y_shffld = np.array([x[1] for x in aux])
    return X_shffld, Y_shffld

# Loads training set
def load_train_features():
    Xtrain = pickle.load(open(os.path.join("raw","features_pretrained_X.pickle"), "rb"))
    Ytrain = pickle.load(open(os.path.join("raw","features_pretrained_Y.pickle"), "rb"))
    #return shuffle_input(Xtrain, Ytrain)
    return Xtrain, Ytrain

# Loads test set
def load_test_features():
    Xtest = pickle.load(open(os.path.join("raw","features_pretrained_test_X.pickle"), "rb"))
    Ytest = pickle.load(open(os.path.join("raw","features_pretrained_test_Y.pickle"), "rb"))
    #return shuffle_input(Xtest, Ytest)
    return Xtest, Ytest


def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


Xtrain, Ytrain = load_train_features()
Xtest, Ytest = load_test_features()

# Comment this line to evaluate only
history = model.fit(Xtrain, Ytrain, batch_size=32, epochs=30, verbose=1, validation_data=(Xtest, Ytest), shuffle=True)
plot(history)

res = model.evaluate(Xtest, Ytest, verbose=1)
print("Loss:", res[0], "\nAccuracy:", res[1])


print("Saving...")
load_save.save_model(model, model_filename)
