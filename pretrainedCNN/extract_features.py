import sys
import os
import pickle
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (240,800,3)
num_classes = 23

# Create the base pre-trained model without last layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)

# Add a pooling layer to the last convolutional output
# TODO: test code, before it was global average pooling without flattening
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(base_model.layers[-1].output)
x = keras.layers.Flatten()(x)
base_model = Model(base_model.input, x)




# - Training set -
#Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtrain,Ytrain,img_shape,labels_ids = dbloader.load_trainingset(img_shape)
# Extract features for training set
Xtrain_feat = base_model.predict(Xtrain, verbose=1)

pickle.dump(Xtrain_feat, open(os.path.join('raw','features_pretrained_X.pickle'), 'wb'))
pickle.dump(Ytrain[()], open(os.path.join('raw','features_pretrained_Y.pickle'), 'wb'))



# - Test set -
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)
# Ectract features for test set
Xtest_feat = base_model.predict(Xtest, verbose=1)

pickle.dump(Xtest_feat, open(os.path.join('raw','features_pretrained_test_X.pickle'), 'wb'))
pickle.dump(Ytest[()], open(os.path.join('raw','features_pretrained_test_Y.pickle'), 'wb'))

