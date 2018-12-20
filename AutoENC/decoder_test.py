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

path = join("raw","encoder.h5")

model = load_save.load_model(path)
if not model:
    raise Exception("Missing encoder")

originals = Xtest[()]
predicted = model.predict(originals)

original_path = join("raw","visualization","originals")
decoded_path = join("raw","visualization","decoded")
for i in range(20):
    k = random.randint(0,len(Xtest)-1)
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

    




