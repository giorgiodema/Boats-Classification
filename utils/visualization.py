
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




def print_filters_output(model,path, X, max_images=10):
    if not os.path.exists(path):
        raise Exception("specified path does not exists")
    # plotting model structure
    plot_model(model, to_file=join(path,'model.png'),show_layer_names=False,show_shapes=True,rankdir='LR')

    for i in range(1,len(model.layers)):

        layer = model.layers[i]
        if len(layer.output_shape)!=4:
            continue

        m = Model(inputs=model.layers[0].output,outputs=layer.output)
        dim = min(max_images,len(X))
        imgs = m.predict(X[:dim])
        if imgs.dtype=='float32': imgs = imgs * 255
        imgs = imgs.astype(np.int32)
        
        
        layer_name = "Layer" + str(i)
        layer_path = join(path,layer_name)
        os.mkdir(layer_path)

        for f in range(layer.output_shape[3]):
            filter_path = join(layer_path, "filter"+str(f))
            os.mkdir(filter_path)
            for i in range(dim):
                img = imgs[i]
                flt = img[:,:,f]
                img_path = join(filter_path,str(i)+".png")
                plt.imshow(flt)
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
print_filters_output(model,save_path,Xtest[()])
