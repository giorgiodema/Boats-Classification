import tflearn
import os
import h5py
import pickle
import re
import PIL
import numpy as np
from os.path import join

train_path = join("raw","trainingset.h5")
test_path = join("raw","testset.h5")

def load_trainingset(img_shape):
    
    labels_ids = {}
    ids_labels = {}

    if not os.path.exists(train_path):
        print("Creating trainingset...")
        # bug in build_dhf5_image comment (shape argument requires (width,height))
        tflearn.data_utils.build_hdf5_image_dataset("dataset\ARGOStraining", (img_shape[1],img_shape[0]), output_path=train_path,
                                mode='folder', categorical_labels=True,
                                normalize=True, grayscale=False,
                                files_extension=None, chunks=False)
    if not os.path.exists(join("raw","ids_labels.pkl")):
        categories = os.listdir(join("dataset\ARGOStraining", ""))
        categories.remove("DBinfo.txt")
        categories.sort()
        ids = [x for x in range(len(categories))]
        labels_ids = {k:v for (k,v) in zip(categories,ids)}
        ids_labels = {k:v for (k,v) in zip(ids,categories)}

        with open(join("raw","ids_labels.pkl"),"wb") as f:
            pickle.dump(ids_labels,f)
        with open(join("raw","labels_ids.pkl"),"wb") as f:
            pickle.dump(labels_ids,f)
    
    print("Loading trainingset ...")
    with open(join("raw","ids_labels.pkl"),"rb") as f:
        ids_labels = pickle.load(f)
    with open(join("raw","labels_ids.pkl"),"rb") as f:
        labels_ids = pickle.load(f)
    h5f = h5py.File(train_path, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return [X,Y,img_shape,ids_labels,labels_ids]

def load_testset(img_shape,labels_ids):

    if not os.path.exists(test_path):
        print("Creating testset...")
        with open(join("dataset","ARGOStest","ground_truth.txt"),"r") as f:
            aux = f.read().split('\n')
            aux = list(filter(lambda x: re.match(r'.*;.*',x),aux))
            aux = list(map(lambda x: (x.split(';')[0],x.split(';')[1].replace(' ','').replace(':','')),aux))
            aux = list(filter(lambda x:x[1] in labels_ids.keys(),aux))
            ground = {k:v for (k,v) in aux}

        dataset = h5py.File(test_path, 'w')
        d_imgshape = (len(ground),img_shape[0],img_shape[1],img_shape[2])
        d_labelshape = (len(ground),len(labels_ids))
        dataset.create_dataset('X', d_imgshape)
        dataset.create_dataset('Y', d_labelshape)

        paths = list(ground.keys())
        for i in range(len(paths)):
            img = PIL.Image.open(join("dataset","ARGOStest","")+paths[i])
            width, height = img.size
            if width != img_shape[1] or height != img_shape[0]:
                img = img.resize((img_shape[1], img_shape[0]))
            if img.mode == 'L':
                img.convert_color('RGB')
            img.load()
            dataset['X'][i] = np.asarray(img, dtype="float32") / 255
            y = np.zeros(shape = d_labelshape[1])
            y[labels_ids[  ground[paths[i]]  ]] = 1
            dataset['Y'][i] = y
            #bp()
    print("Loading testset ...")
    h5f = h5py.File(test_path, 'r')
    X = h5f['X']
    Y = h5f['Y']

    return [X,Y]