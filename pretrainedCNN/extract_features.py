import sys
import os
import pickle
from pdb import set_trace
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (240,800,3)
num_classes = 24

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)

x = GlobalAveragePooling2D()(base_model.layers[-1])
base_model = Model(base_model.input, x)

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)


Xtrain_feat = base_model.predict(Xtrain, verbose=1)


pickle.dump(Xtrain_feat[:2000], open('features_pretrained_0_2000.pickle', 'wb'))
pickle.dump(Xtrain_feat[2000:], open('features_pretrained_2000_end.pickle', 'wb'))
pickle.dump(Ytrain[()], open('features_pretrained_Y.pickle', 'wb'))
del Xtrain_feat
'''
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

Xtest_feat = base_model.predict(Xtest, verbose=1)

pickle.dump(Xtest_feat, open('features_pretrained_test.pickle', 'wb'))
pickle.dump(Ytest[()], open('features_pretrained_test_Y.pickle', 'wb'))
'''
