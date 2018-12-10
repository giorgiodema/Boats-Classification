import dbloader


Xtrain,Ytrain,img_shape,ids_labels,labels_ids = dbloader.load_trainingset((800,240,3))
X,Y = dbloader.load_testset((800,240,3),labels_ids)

print("Hello")