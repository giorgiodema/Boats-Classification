import cnnmodel
import numpy as np
import os
import sys
from os.path import join
sys.path.append(os.path.abspath("utils"))
import dbloader

path = join("raw","CNN.pkl")
img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

print("Loading cnn...")
cnn = cnnmodel.load_classifier(path)
if not cnn:
    print("Creating cnn...")
    cnn = cnnmodel.ConvolutionalNN(img_shape,num_classes)
    print("Training cnn...")
    try:
        cnn.fit(Xtrain[()],Ytrain[()],Xtest[()],Ytest[()])
    except KeyboardInterrupt:
        pass
    print("Saving classifier...")
    cnnmodel.save_classifier(path,cnn)
    

