import os
from os.path import join
import sys
import numpy as np
import enc_classifier
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

model_path = join("raw","encoder02.h5")

model = load_save.load_model(model_path)
if not model:
    raise Exception("No model...")

originals = Xtrain[30:40]
predicted = model.predict(originals)

for i in range(originals.shape[0]):
    o = originals[i] * 255
    p = predicted[i] * 255
    o = o.astype(np.int32)
    p = p.astype(np.int32)
    plt.imshow(o)
    plt.show()
    plt.imshow(p)
    plt.show()

    




