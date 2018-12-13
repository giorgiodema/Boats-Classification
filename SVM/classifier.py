import sys
import os
import pickle
from pdb import set_trace
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten
from keras import backend as K
import sklearn.svm
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader


def save_classifier(path,clf):
    with open(path,"wb") as f:
        pickle.dump(clf,f)

def load_classifier(path):
    clf = None
    if os.path.exists(path):
        with open(path,"rb") as f:
            clf = pickle.load(f)
    return clf

class SVMwithFeatures:

    def __init__(self,img_shape):
        self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
        x1 = MaxPooling2D(pool_size=(2,2), strides=2)(self.model.layers[-1].output)
        x2 = Flatten()(x1)        
        self.model = Model(self.model.input,x2)

        self.clf = sklearn.svm.LinearSVC(max_iter=100000)

    
    def fit(self,X,y):
        Xf = self.__extract_features(X)
        self.clf.fit(Xf,y)

    def __extract_features(self,X):
        X_feat = self.model.predict(X,verbose=1)
        return X_feat
    
    def predict(self,X):
        Xf = self.__extract_features(X)
        y = self.clf.predict(Xf)
        return y

