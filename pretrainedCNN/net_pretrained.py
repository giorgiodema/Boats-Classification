import sys
import os
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save


img_shape = (800,240,3)
num_classes = 23
model_filename = "inceptionV3_pretrained.h5"

model = load_save.load_model(model_filename)
if not model:
   # create the base pre-trained model
   base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(800,240,3))

   # add a global spatial average pooling layer
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   # let's add a fully-connected layer
   x = Dense(1024, activation='relu')(x)
   # and a logistic layer -- let's say we have 200 classes
   predictions = Dense(num_classes, activation='softmax')(x)

   # this is the model we will train
   model = Model(inputs=base_model.input, outputs=predictions)

   # first: train only the top layers (which were randomly initialized)
   # i.e. freeze all convolutional InceptionV3 layers
   for layer in base_model.layers:
      layer.trainable = False

   # compile the model (should be done *after* setting layers to non-trainable)
   model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()


Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)


try:
    model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1, validation_data = (Xtest,Ytest), shuffle="batch")
except KeyboardInterrupt:
    # Save the model
    print("Saving...")
    load_save.save_model(model, model_filename)
    quit()
