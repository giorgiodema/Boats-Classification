import dbloader


Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset()
X,Y = dbloader.load_testset(labels_ids)

print("Hello")