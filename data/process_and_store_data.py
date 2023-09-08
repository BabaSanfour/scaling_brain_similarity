import os
import sys
import json
import h5py
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
import torchvision
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_data_config_parser


import logging

proj_path = os.path.dirname(os.path.abspath(__file__))
# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load CSV data
csv_data = None #pd.read_csv("LOC_val_solution.csv")


def store_many_hdf5(images: np.array, labels: np.array, folder: str, hdf5_dir: str) -> None:
    """Stores an array of images to HDF5.

    Args:
    images: np.ndarray
        Images array, shape (N, 224, 224, 3), to be stored.
    labels: np.ndarray
        Labels array, shape (N,), to be stored.
    folder: str
        Folder name for the HDF5 file.
    hdf5_dir: str
        Folder to save HDF5 files.

    """
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    # Create a new HDF5 file
    file = h5py.File(os.path.join(hdf5_dir, f"{folder}.h5"), "w")
    logger.info(f"{folder} h5 file created")

    # Create datasets in the file
    dataset = file.create_dataset("images", np.shape(images), data=images)
    metaset = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()
    logger.info(f"{folder} h5 is ready")


def make_array(data_dir: str, data: str, dataset: str) -> tuple:
    """
    Create arrays of images and labels based on the specified analysis type and folder.

    Args:
        data_dir (str): The path to the data directory.
        data (str):     The data type ('train imagenet or places365' or 'val imagenet or places365').
        dataset (str):   The dataset name ('imagenet' or 'places365').

    Returns:
        tuple: A tuple containing image and label arrays.
    """
    # Concatenate array of images
    img_array = []
    label_array = []

    # Resize images to 224x224
    resize = torchvision.transforms.Resize((224, 224))

        # Load label matching data
    with open(os.path.join(proj_path, f"match_labels_{dataset}.json"), 'r') as f:
        match_labels = json.load(f)


    if data in ['train_imagenet', 'train_places365', 'val_places365']:
        label_folders = sorted(
            [
                label
                for label in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, label))
            ]
        )
        for label_folder in label_folders:
            label = match_labels.get(label_folder, None)
            pictures_paths = sorted(
                [
                    os.path.join(os.path.join(data_dir, label_folder), sname)
                    for sname in os.listdir(os.path.join(data_dir, label_folder))
                ]
            )

            for picture_path in tqdm(pictures_paths):
                img_sample = cv2.imread(picture_path)  # Read picture
                if img_sample is None:
                    continue
                img_sample = img_sample[:, :, ::-1]  # Transform from BGR to RGB
                PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
                sample = np.asarray(resize(PIL_image), dtype=np.uint8)
                img_array.append(sample)
                label_array.append(label)
    
    elif data == 'val_imagenet':
        pictures_paths = sorted(
            [
                os.path.join(data_dir, sname)
                for sname in os.listdir(data_dir)
            ]
        )

        for picture_path in pictures_paths:
            img_sample = cv2.imread(picture_path)  # Read picture
            if img_sample is None:
                continue
            # Extract labels from a CSV file (assuming a specific format)
            name = os.path.splitext(os.path.basename(picture_path))[0]
            label = csv_data.loc[csv_data['ImageId'] == name, 'PredictionString']
            label = match_labels.get(label, None)  # Use match_labels if needed
            img_sample = img_sample[:, :, ::-1]  # Transform from BGR to RGB
            PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
            sample = np.asarray(resize(PIL_image), dtype=np.uint8)
            img_array.append(sample)
            label_array.append(label)

    logger.info(f"Image array shape: {np.asarray(img_array).shape}")
    logger.info(f"Label array shape: {np.asarray(label_array).shape}")
    return np.asarray(img_array), np.asarray(label_array)


if __name__ == '__main__':

    parser = get_data_config_parser()
    args = parser.parse_args()
    hdf5_dir = os.path.join(args.hdf5_dir, args.dataset)

    for scale in range(1, 11):
        folder = f"scaling_fac_{scale}"
        logger.info(f"Processing data for {folder}")
        begin_time = datetime.datetime.now()
        img_array, label_array = make_array(folder, f"train_{args.dataset}", args.dataset)
        store_many_hdf5(img_array, label_array, folder, hdf5_dir)
        logger.info(f"Processing time for {folder}: {datetime.datetime.now() - begin_time}")

    begin_time = datetime.datetime.now()
    logger.info(f"Processing data for validation set")
    if args.dataset == "imagenet":
        img_array, label_array = make_array(os.path.join('ILSVRC', 'Data', 'CLS-LOC', 'val'), "val_imagenet", args.dataset)
        store_many_hdf5(img_array, label_array, "valid", hdf5_dir)
    elif args.dataset == "places365":
        img_array, label_array = make_array(os.path.join('places365_standard', 'val'), "val_places365", args.dataset)
        store_many_hdf5(img_array, label_array, "valid", hdf5_dir)

    logger.info(f"Processing time for validation set: {datetime.datetime.now() - begin_time}")
