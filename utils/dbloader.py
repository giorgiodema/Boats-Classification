import tflearn
import os
import h5py
import pickle
import re

def load_trainingset():
    
    img_shape = (800,240,3)
    labels_ids = {}
    ids_labels = {}

    print("Creating trainingset...")
    if not os.path.exists("trainingset.h5"):
        tflearn.data_utils.build_hdf5_image_dataset("ARGOStraining", img_shape, output_path='trainingset.h5',
                                mode='folder', categorical_labels=True,
                                normalize=True, grayscale=False,
                                files_extension=None, chunks=False)
    if not os.path.exists("raw\ids_labels.pkl"):
        categories = os.listdir("ARGOStraining\\")
        categories.remove("DBinfo.txt")
        categories.sort()
        ids = [x for x in range(len(categories))]
        labels_ids = {k:v for (k,v) in zip(categories,ids)}
        ids_labels = {k:v for (k,v) in zip(ids,categories)}

        with open("raw\ids_labels.pkl","wb") as f:
            pickle.dump(ids_labels,f)
        with open("raw\labels_ids.pkl","wb") as f:
            pickle.dump(labels_ids,f)
    
    print("Loading trainingset ...")
    with open("raw\ids_labels.pkl","rb") as f:
        ids_labels = pickle.load(f)
    with open("raw\labels_ids.pkl","rb") as f:
        labels_ids = pickle.load(f)
    h5f = h5py.File('trainingset.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']

    return [X,Y,img_shape,ids_labels,labels_ids]

def load_testset(labels_ids):

    img_shape = (800,240,3)

    if not os.path.exists("testset.h5"):
        print("Creating testset...")
        with open("ARGOStest\ground_truth.txt","r") as f:
            aux = f.read().split('\n')
            aux = list(filter(lambda x: re.match(r'.*;.*',x),aux))
            aux = list(map(lambda x: (x.split(';')[0],x.split(';')[1].replace(' ','').replace(':','')),aux))
            ground = {k:v for (k,v) in aux}
        with open("raw\img-id.txt","w") as f:
            for img in ground.keys():
                if ground[img] in labels_ids.keys():
                    f.write(".\ARGOStest\{} {}\n".format(img,labels_ids[ground[img]]))

        tflearn.data_utils.build_hdf5_image_dataset("raw\img-id.txt", img_shape, output_path='testset.h5',
                             mode='file', categorical_labels=True,
                             normalize=False, grayscale=False,
                             files_extension=['.jpg'], chunks=False)


    print("Loading testset ...")
    h5f = h5py.File('testset.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']

    return [X,Y]