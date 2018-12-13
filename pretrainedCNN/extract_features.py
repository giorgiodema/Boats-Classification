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

# create the base pre-trained model without last layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)

# add a pooling layer to the last convolutional output
x = GlobalAveragePooling2D()(base_model.layers[-1].output)
base_model = Model(base_model.input, x)


# Extract features for training set
Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)

Xtrain_feat = base_model.predict(Xtrain, verbose=1)

pickle.dump(Xtrain_feat, open('features_pretrained_X.pickle', 'wb'))
pickle.dump(Ytrain[()], open('features_pretrained_Y.pickle', 'wb'))



# Ectract features for test set
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

Xtest_feat = base_model.predict(Xtest, verbose=1)

pickle.dump(Xtest_feat, open('features_pretrained_test_X.pickle', 'wb'))
pickle.dump(Ytest[()], open('features_pretrained_test_Y.pickle', 'wb'))

