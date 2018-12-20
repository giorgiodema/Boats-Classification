import keras
import sys
import os
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
model_filename = "LeNet.h5"



model = load_save.load_model(model_filename)
if not model:
    model = lenet.LeNet(img_shape,num_classes)

model.summary()


Xtrain,Ytrain,img_shape,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

#model.evaluate(Xtest,Ytest)
#quit()

try:
    #model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1, validation_data = (Xtest,Ytest), shuffle=True)
    history = model.fit(Xtrain, Ytrain, batch_size=32, epochs=30, verbose=1, validation_data=(Xtest,Ytest), shuffle="batch")
except KeyboardInterrupt:
    # Save the model
    print("Saving...")
    load_save.save_model(model, model_filename)
    quit()
print("Training Completed...")
print("Saving...")
load_save.save_model(model, model_filename)
print("Done...")
    


