
import os
import pickle
import numpy as np
from pprint import pprint
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


print(" - SVM - loading data")
Xtrain = pickle.load(open(os.path.join("raw","features_pretrained_X.pickle"), "rb"))
Ytrain = pickle.load(open(os.path.join("raw","features_pretrained_Y.pickle"), "rb"))

Xtest = pickle.load(open(os.path.join("raw","features_pretrained_test_X.pickle"), "rb"))
Ytest = pickle.load(open(os.path.join("raw","features_pretrained_test_Y.pickle"), "rb"))

assert(Xtrain[0].shape == Xtest[0].shape)
assert(Ytrain[0].shape == Ytest[0].shape)



def preprocess_y(y):
	y = np.asarray(y)
	i, j = np.where(y == 1)
	print(i)
	return j


print(" - SVM - training")
#clf = LinearSVC(verbose=1)
#clf.fit(Xtrain, preprocess_y(Ytrain))
#joblib.dump(clf, os.path.join("SVM_spadala","svm_clf.joblib"))
clf = joblib.load(os.path.join("SVM_spadala","svm_clf.joblib"))


print(" - SVM - evaluating")
y_predicted = clf.predict(Xtest)



print(" - SVM - confusion matrix")
print("shape:", (Ytrain.shape[1], Ytrain.shape[1]))
conf_matrix = [[0 for j in range(Ytrain.shape[1])] for i in range(Ytrain.shape[1])]
np.zeros((Ytrain.shape[1], Ytrain.shape[1]))
for y_pred, Y_gt in zip(y_predicted, preprocess_y(Ytest)):
	conf_matrix[Y_gt][y_pred] += 1
#conf_matrix /= len(Ytest)


#pprint(conf_matrix)
for r in conf_matrix:
	for n in r:
		print(str(n)+'\t',end='')
	print()
