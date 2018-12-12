import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import sys
'''if sys.platform == "linux":
    sys.path.append(os.path.abspath("utils"))
else:    
    path = os.getcwd()+"\\utils"
    sys.path.append(path)'''
sys.path.append(os.path.abspath("utils"))
import dbloader
import load_save
import alexnet
import lenet

img_shape = (240,800,3)
num_classes = 23
model_filename = "inceptionV3.h5"



model = load_save.load_model(model_filename)
if not model:
    x = keras.layers.Input(shape = img_shape)
    #model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=img_shape, alpha=1.0, depth_multiplier=1, include_top=True, weights=None, input_tensor=x, pooling=None, classes=num_classes)
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=x, input_shape=img_shape, pooling=None, classes=num_classes)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    #model = alexnet.AlexNet(img_shape,num_classes)
    #model = lenet.LeNet(img_shape,num_classes)

model.summary()


Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)


try:
    #model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1, validation_data = (Xtest,Ytest), shuffle=True)
    model.fit(  Xtrain, Ytrain, batch_size=16, epochs=20, verbose=1, shuffle="batch", callbacks=[keras.callbacks.CSVLogger("logger.csv", separator=',', append=False)/
                keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
except KeyboardInterrupt:
    # Save the model
    #print("Saving...")
    #load_save.save_model(model, model_filename)
    quit()
print("Training Completed...")
#print("Saving...")
#load_save.save_model(model, model_filename)
#print("Done...")
    


