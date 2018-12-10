import keras
import sys
import load_save
import os
path = os.getcwd()+"\\utils"
sys.path.append(path)
import dbloader

img_shape = (800,240,3)
num_classes = 23



model = load_save.load_model()
if not model:
    x = keras.layers.Input(shape = img_shape)
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=x, input_shape=img_shape, pooling=None, classes=num_classes)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

model.summary()


Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)


try:
    model.fit(Xtrain, Ytrain, batch_size=64, epochs=100, verbose=1, validation_data = (Xtest,Ytest), shuffle="batch")
except KeyboardInterrupt:
    pass
    
# Save the model
load_save.save_model(model)

