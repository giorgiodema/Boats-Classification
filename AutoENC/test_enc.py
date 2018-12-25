import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from os.path import join
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import enc_CNN
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save



img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

model_path = os.path.join("raw","encoder.h5")
enc = load_save.load_model(model_path)

originals = Xtest[:20]
predicted = enc.predict(originals)

originals_path = os.path.join("FiltersImages","Encoder","Originals")
predicted_path = os.path.join("FiltersImages","Encoder","Predicted")
predicted = enc.predict(originals)
for i in range(len(predicted)):
    img_o = originals[i] *255
    img_p = predicted[i] * 255
    img_o = np.ndarray.astype(img_o,np.int32)
    img_p = np.ndarray.astype(img_p,np.int32)
    img_o_path = os.path.join(originals_path,str(i)+".png")
    img_p_path = os.path.join(predicted_path,str(i)+".png")
    plt.imshow(img_o)
    plt.savefig(img_o_path)
    plt.imshow(img_p)
    plt.savefig(img_p_path)



