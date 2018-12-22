import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from os.path import join
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import enc_classifier
sys.path.append(os.path.abspath("utils"))
import dbloader


img_shape = (240,800,3)
num_classes = 24

Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset(img_shape)
Xtest,Ytest = dbloader.load_testset(img_shape,labels_ids)

print("formatting ground vector to categorical for SVM...")
ytrain = np.zeros(shape = Ytrain.shape[0])
for i in range(Ytrain.shape[0]):
    y = Ytrain[i]
    for k in range(y.shape[0]):
        if y[k] == 1:
            ytrain[i] = k
            break

ytest = np.zeros(shape = Ytest.shape[0])
for i in range(Ytest.shape[0]):
    y = Ytest[i]
    for k in range(y.shape[0]):
        if y[k] == 1:
            ytest[i] = k
            break


clf = enc_classifier.AutoEncSVMclassifier(img_shape,num_classes)
clf.fit(Xtrain[()],ytrain,Xtest)

print("Predicting...")
y_pred = clf.predict(Xtest[()])

# STATISTICS
cm = np.zeros(shape=(len(ids_labels),len(ids_labels)),dtype=np.int32)
for i in range(len(y_pred)):
    ground = int(ytest[i])
    guess = int(y_pred[i])
    cm[ground,guess] +=1


classes = []
for i in range(len(ids_labels)):
        classes.append(ids_labels[i])



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
plt.savefig("autoenc_cm.png")

print("Accuracy:")
for i in range(len(classes)):
    print("{:50s}:{:3f}".format(ids_labels[i],classes_accuracy[i]))

print("{:50s}:{:3f}".format("GLOBAL ACCURACY",global_accuracy))