
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
import keras




def print_layer_output(model,path, X, rescale=True, max_images=10):
    if not os.path.exists(path):
        raise Exception("specified path does not exists")
    # plotting model structure
    #plot_model(model, to_file=join(path,'model.png'),show_layer_names=False,show_shapes=True,rankdir='LR')

    for i in range(1,len(model.layers)):

        layer = model.layers[i]
        if len(layer.output_shape)!=4:
            continue

        m = Model(inputs=model.layers[0].output,outputs=layer.output)
        dim = min(max_images,len(X))
        imgs = m.predict(X[:dim])
        imgs = imgs.astype(np.int32)
        if rescale: imgs = imgs * 255
        
        layer_name = "Layer" + str(i)
        layer_path = join(path,layer_name)
        os.mkdir(layer_path)

        for f in range(layer.output_shape[3]):
            filter_path = join(layer_path, "filter"+str(f))
            os.mkdir(filter_path)
            for i in range(dim):
                img = imgs[i]
                img_path = join(filter_path,str(i)+".png")
                plt.imshow(img)
                plt.savefig(img_path)

            


        






model_path = join("raw","CNN.h5")
model = load_save.load_model(model_path)
if not model:
    x = Input(shape=(240,800,3))
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=x, input_shape=None, pooling=None, classes=1000)
save_path = join("raw","CNN")

img_shape=(240,800,3)
Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset((800,240,3))
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)
print_layer_output(model,save_path,Xtest[()])
"""
img_shape = (240,800,3)
path = join("raw","CNN.pkl")
plt_path = 'raw'

layer_to_print = 2

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
    #k = random.randint(0,len(Xtest)-1)
    k = i + 20
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


print("ciao")
"""