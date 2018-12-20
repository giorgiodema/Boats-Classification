
import os
import pickle
import numpy as np
from pprint import pprint
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


print(" - SVM - loading data")
Xtrain = pickle.load(open("features_pretrained_X.pickle", "rb"))
Ytrain = pickle.load(open("features_pretrained_Y.pickle", "rb"))

Xtest = pickle.load(open("features_pretrained_test_X.pickle", "rb"))
Ytest = pickle.load(open("features_pretrained_test_Y.pickle", "rb"))

assert(Xtrain[0].shape == Xtest[0].shape)
assert(Ytrain[0].shape == Ytest[0].shape)



def preprocess_y(y):
	y = np.asarray(y)
	i, j = np.where(y == 1)
	print(i)
	return j


print(" - SVM - training")
clf = LinearSVC(verbose=1)
clf.fit(Xtrain, preprocess(Ytrain))
joblib.dump(clf, os.path.join("SVM_spadala","svm_clf.joblib"))



print(" - SVM - evaluating")
y_predicted = clf.predict(Xtest)



print(" - SVM - confusion matrix")
print("shape:", (Xtrain.shape[1], Xtrain.shape[1]))
mat = np.zeros((Xtrain.shape[1], Xtrain.shape[1]))
for y_pred, Y_gt in zip(y_predicted, preprocess(Ytest)):
	mat[Y_gt][y_pred] += 1
mat /= len(Ytest)

pprint(mat)
