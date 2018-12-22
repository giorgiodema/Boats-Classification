import net_pretrained
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from os.path import join
sys.path.append(os.path.abspath("utils"))
import dbloader


img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)


print("Creating cnn...")
cnn = net_pretrained.pretrainedCNN(img_shape,num_classes)
print("Training cnn...")
cnn.fit(Xtrain[()],Ytrain[()],Xtest[()],Ytest[()])

    

# from hot encoding to ordinal variables
Ypred = cnn.predict(Xtest[()])
ypred = np.zeros(shape = Ypred.shape[0],dtype=np.int32)
ytest = np.zeros(shape = Ypred.shape[0],dtype=np.int32)
for i in range(Ypred.shape[0]):
    tr = Ytest[()][i]
    pr = Ypred[i]
    trindex = np.where(tr==1)[0]
    prindex = np.where(pr==pr.max())[0]
    ytest[i] = trindex
    ypred[i] = prindex

classes = []
for i in range(len(ids_labels)):
        classes.append(ids_labels[i])


cm = np.zeros(shape=(len(classes),len(classes)),dtype=np.int32)
for i in range(ypred.shape[0]):
    cm[ytest[i],ypred[i]]+=1

#classes accuracy
classes_accuracy = np.ndarray(shape=(len(classes)))
for i in range(classes_accuracy.shape[0]):
    axis_sum = np.sum(cm[i])
    if axis_sum==0:
        classes_accuracy[i] = -1
    else:
        classes_accuracy[i] = cm[i,i] / axis_sum

#global accuracy
diagonal = 0
for i in range(len(classes)):
    diagonal += cm[i,i]
global_accuracy = diagonal / np.sum(cm,axis=(0,1))

#normalization
cm = cm.astype(np.float32)
for x in range(len(classes)):
    for y in range(len(classes)):
        axis_sum = np.sum(cm[x])
        axis_sum = axis_sum if axis_sum!=0 else 1
        cm[x,y] = cm[x,y] / axis_sum
        

plt.matshow(cm)
plt.colorbar()
plt.savefig("pretrained_cm.png")

print("Accuracy:")
for i in range(len(classes)):
    print("{:50s}:{:3f}".format(ids_labels[i],classes_accuracy[i]))

print("{:50s}:{:3f}".format("GLOBAL ACCURACY",global_accuracy))