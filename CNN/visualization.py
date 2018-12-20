import cnnmodel
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from os.path import join
from matplotlib import pyplot as plt
import random
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save

img_shape = (240,800,3)
path = join("raw","CNN.pkl")
plt_path = 'raw'

layer_to_print = 5

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

print("Loading cnn...")
cnn = cnnmodel.load_classifier(path).model
if not cnn:
    raise Exception("No model to visualize")

plot_model(cnn, to_file='model.png',show_layer_names=False,show_shapes=True,rankdir='LR')

inputl = cnn.layers[0]
conv01 = cnn.layers[layer_to_print]
conv_01_model = Model(inputs=inputl.output,outputs=conv01.output)

originals = Xtest[()]
predicted = conv_01_model.predict(originals)

filters_path = join("raw","visualization","CNNfilters","Layer"+str(layer_to_print))
originals_path=join("raw","visualization","CNNfilters","Layer"+str(layer_to_print),"originals")
if not os.path.exists(filters_path):
    os.makedirs(filters_path)
    os.makedirs(originals_path)
for i in range(20):
    fig_name = str(i) + ".png"
    k = random.randint(0,len(Xtest)-1)

    # printing original image
    o_path = join(originals_path,fig_name)
    o = originals[k] * 255
    o = o.astype(np.int32)
    plt.imshow(o)
    plt.savefig(o_path)

    #printing filters
    filt = predicted[k]
    for c in range(filt.shape[2]):
        f = filt[:, :, c] *255
        f = f.astype(np.int32)
        f_path=join(filters_path,str(c))
        p = join(filters_path,str(c),fig_name)
        if not os.path.exists(f_path): os.mkdir(f_path)
        plt.imshow(f)
        plt.savefig(p)

    """
    fig_name = str(i) + ".png"
    o_path = join(original_path,fig_name)
    p_path = join(decoded_path,fig_name)

    o = originals[k] * 255
    p = predicted[k] * 255
    o = o.astype(np.int32)
    p = p.astype(np.int32)
    plt.imshow(o)
    plt.savefig(o_path)
    plt.imshow(p)
    plt.savefig(p_path)
    """
print("ciao")

