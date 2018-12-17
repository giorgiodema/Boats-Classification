import os
from os.path import join
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
import enc_classifier
sys.path.append(os.path.abspath("utils"))
import dbloader



path = join("raw","enc_clf.pkl")
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

print("Loading classifier...")
clf = enc_classifier.load_classifier(path)
if not clf:
    clf = enc_classifier.AutoEncSVMclassifier(img_shape,num_classes)
    clf.fit(Xtrain[()],y,Xtest)
    print("Saving classifier...")
    enc_classifier.save_classifier(path,clf)

print("Predicting...")
y_pred = clf.predict(Xtest[()])

# STATISTICS
conf_matrix = np.zeros(shape=(len(ids_labels),len(ids_labels)),dtype=np.int32)
for i in range(len(y_pred)):
    ground = int(ytest[i])
    guess = int(y_pred[i])
    conf_matrix[ground,guess] +=1

right_pred = 0
miss_pred = 0
for i in range(len(ids_labels)):
    for j in range(len(ids_labels)):
        if i==j: right_pred += conf_matrix[i,j]
        else: miss_pred += conf_matrix[i,j]

print("Confusion matrix (x-axis = predictions, y-axis = ground")
print("-------------------------------------------------------")
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        print("%03s"%conf_matrix[i,j],end=" ")
    print("\n")
print("accuracy = {}".format(right_pred/(right_pred+miss_pred)))