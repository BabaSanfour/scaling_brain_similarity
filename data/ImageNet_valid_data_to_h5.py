"""
    For each set file:
    - Create a HDF5 file with pictures and their IDs.
    ---------------
    Output Files:
    ---------------
    HDF5 file for validation set
    Parameters:
    images       images array, (N, 224, 224, 3) to be stored
    labels       labels array, (N, ) to be stored

"""

import os
import pandas as pd
import cv2
import h5py
import json

import datetime
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image

def store_many_hdf5(images, labels, folder):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 224, 224, 1) to be stored
        labels       labels array, (N, ) to be stored
    """
    hdf5_dir = "/home/hamza97/scratch/data/scaling_data/hdf5/"
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    # Create a new HDF5 file
    file = h5py.File("%s%s.h5"%(hdf5_dir,folder), "w")
    print("{} h5 file created".format(folder))

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), data=images) #h5py.h5t.STD_U8BE,
    metaset = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()
    print("{} h5 is ready".format(folder))

def make_array(dir, csv_data, match_labels):
    # Concatenate array of images
    img_array = []
    label_array = []

    # resize images to 224 224
    resize = torchvision.transforms.Resize((224, 224))

    pictures_pathes = sorted(
        [
            os.path.join(dir, sname)
            for sname in os.listdir(dir)
        ]
    )
    # run through the pictures list
    pictures_loop_generator = tqdm(pictures_pathes)
    for  picture in pictures_loop_generator:
        img_sample = cv2.imread(picture) # read picture
        name = os.path.splitext(os.path.basename(picture))[0]
        id = csv_data.loc[csv_data['ImageId'] == name, 'PredictionString'].iloc[0][:9]
        id = match_labels[id]
        if img_sample is None:
            continue
        img_sample = img_sample[:,:,::-1] # transform from BGR 2 RGB
        # transform the image
        PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
        sample = np.asarray(resize(PIL_image), dtype=np.uint8)
        # append image and label to image and labels list
        img_array.append(sample)
        label_array.append(id)

    # print label and image array shapes
    print(np.asarray(label_array).shape)
    print(np.asarray(img_array).shape)
    # return image and label arrays
    return np.asarray(img_array), np.asarray(label_array)

if __name__ == '__main__':
    csv_data = pd.read_csv("LOC_val_solution.csv")
    with open('match_labels.json', 'r') as f:
        match_labels = json.load(f)
    begin_time = datetime.datetime.now()
    img_array, label_array = make_array("val", csv_data, match_labels)
    store_many_hdf5(img_array,label_array, "val")
    print(datetime.datetime.now()-begin_time)
