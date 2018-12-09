import tflearn
import os
import h5py
import pickle
import re



def load_testset():

    img_shape = (800,240,3)

    if not os.path.exists("testset.h5"):
        print("Creating testset...")
        with open("ARGOStest\ground_truth.txt","r") as f1:
            l = f1.read().split('\n')
        labels = []
        img_labels = {}
        for i in range(len(l)):
            if not re.match(r'.*.jpg;.*',l[i]): continue
            img,lab = l[i].split(';')
            img_labels[img] = lab
            if lab not in labels: labels.append(lab)
        labels_ids = {}
        ids_labels = {}
        for i in range(len(labels)):
            labels_ids[labels[i]] = i
            ids_labels[i] = labels[i]
        
        ground = open(".\Argostest\ground.txt","w")
        for img in img_labels.keys():
            ground.write("Argostest\{} {}\n".format(img,labels_ids[img_labels[img]]))
        ground.flush()
        ground.close()

        with open(".\ids_labels.pkl","wb") as f2:
            pickle.dump(ids_labels,f2)


        tflearn.data_utils.build_hdf5_image_dataset("ARGOStest\ground.txt", img_shape, output_path='testset.h5',
                             mode='file', categorical_labels=True,
                             normalize=False, grayscale=False,
                             files_extension=['.jpg'], chunks=False)
    print("Loading testgset ...")
    h5f = h5py.File('testset.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']
    with open("ids_labels.pkl","rb") as f3:
        ids_labels = pickle.load(f3)
    return [X,Y,img_shape,len(Y),ids_labels,labels_ids]
