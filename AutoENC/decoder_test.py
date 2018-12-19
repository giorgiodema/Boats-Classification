import os
from os.path import join
import sys
import numpy as np
import enc_classifier
from matplotlib import pyplot as plt
from PIL import Image
import random
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

path = join("raw","enc_clf.pkl")

clf = enc_classifier.load_classifier(path)
if not clf:
    raise Exception("Missing encoder")

originals = Xtrain[0:Xtrain.shape[0]/4]
predicted = clf.encoder.predict(originals)

for i in range(10):
    i = i+random.randint(0,originals.shape[0]-10)
    o = originals[i] * 255
    p = predicted[i] * 255
    o = o.astype(np.int32)
    p = p.astype(np.int32)
    plt.imshow(o)
    plt.show()
    plt.imshow(p)
    plt.show()

    




