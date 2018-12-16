import sys
import os
import pickle
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