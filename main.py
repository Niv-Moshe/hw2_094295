import torch  # root package
from torch.utils.data import Dataset, DataLoader  # dataset representation and loading
from torchvision import datasets, models  # vision datasets,architectures & transforms
import torchvision.transforms as T  # composable transforms
import imgaug as ia
from imgaug import augmenters as iaa
import os
from pathlib import Path
from glob import glob
from collections import Counter
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from scipy.ndimage.interpolation import shift
from skimage.filters import threshold_otsu
from ast import literal_eval

# Set seed
ia.seed(1)

# Create new directory to store cleaned images (copy raw data)
CLEAN_DATA_PATH = 'data_clean'
TRAIN_PATH = f'{CLEAN_DATA_PATH}/train'
VALID_PATH = f'{CLEAN_DATA_PATH}/val'
DATA_PATHS = [TRAIN_PATH, VALID_PATH]

# Remove data_clean folder is exists
if os.path.exists(CLEAN_DATA_PATH):
    shutil.rmtree(CLEAN_DATA_PATH)
    print("Removed data_clean folder")

# Create new directory for cleaned data
shutil.copytree('data', CLEAN_DATA_PATH)
print('Created data_clean folder')


# Rename images files in train and val sets
# def rename_data():
for data_type in DATA_PATHS:
    for folder in os.listdir(data_type):
        for index, file in enumerate(os.listdir(data_type + '/' + folder)):
            if file == '.DS_Store':
                os.remove(os.path.join(data_type, folder, file))
                print("deleted ds_store")
                continue
            data_type_name = data_type.split('/')[-1]
            os.rename(os.path.join(data_type, folder, file), os.path.join(data_type, folder,
                                                                          ''.join(
                                                                              [str(data_type_name), '_',
                                                                               str(folder),
                                                                               '_', str(index), '.png'])))


# Setup function to display random images based on a label (for manual review) - Includes both train and valid
def show_images(label, num):
    images, labels, filenames = [], [], []

    # Add train data images
    data_folders = [TRAIN_PATH, VALID_PATH]

    for data_folder in data_folders:
        for image in os.listdir(data_folder + '/' + label):
            images.append(os.path.join(data_folder, label, image))
            filenames.append(f'{image}')

    plt.figure(1, figsize=(18, 11))
    plt.axis('off')

    n = 0
    for i in range(num):
        n += 1
        random_index = random.choice(range(len(images)))
        random_filename = filenames[random_index]
        random_img = images[random_index]
        img = cv2.imread(random_img, cv2.IMREAD_GRAYSCALE)
        plt.subplot(int(np.sqrt(num)), int(np.sqrt(num)), n)
        plt.axis('off')
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        random_data_type = random_filename.split('/')[0]
        random_img_name = random_filename.split('/')[-1].split('.')[0]
        plt.title(f'{random_img_name}')

    plt.show()


# show_images('iii', 25)

# Setup function to add file into list for deleting later
deleted_imgs = []


def delete_file(data_type, img_name):
    global deleted_imgs
    label = img_name.split('_')[0]
    file_path = f'{CLEAN_DATA_PATH}/{data_type}/{label}/{data_type}_{img_name}.png'

    # Add file path to list to track deleted items
    deleted_imgs = deleted_imgs + [file_path]

    # Remove duplicates in list
    deleted_imgs = list(dict.fromkeys(deleted_imgs))

    # Delete file
    try:
        os.remove(file_path)
        print(f'Successfully deleted {file_path}')
    except:
        pass


# Setup function to move file into another folder
# Datatype = 'train' or 'valid', Imgname (without .png) e.g. 'i_123'
# Arguments should be in this format e.g. 'train', 'i_123', 'iii'


def move_file(data_type, img_name, new_label):
    old_label = img_name.split('_')[0]
    source = f'{CLEAN_DATA_PATH}/{data_type}/{old_label}/{data_type}_{img_name}.png'
    dest = f'{CLEAN_DATA_PATH}/{data_type}/{new_label}/{data_type}_{img_name}.png'

    try:
        shutil.move(source, dest)
        print(f'Successfully moved {source} to {new_label}')
    except:
        pass


# Add images to list for deletion
delete_df = pd.read_csv("delete.csv", converters={'i': literal_eval, 'ii': literal_eval, 'iii': literal_eval,
                                                  'iv': literal_eval, 'v': literal_eval, 'vi': literal_eval,
                                                  'vii': literal_eval, 'viii': literal_eval, 'ix': literal_eval,
                                                  'x': literal_eval})
for col in delete_df.columns:
    for index in delete_df[col][0]:
        delete_file('train', f'{col}_{index}')

print(len(deleted_imgs))
print(deleted_imgs)

# Files to move
move_df = pd.read_csv("move.csv", converters={'i': literal_eval, 'ii': literal_eval, 'iii': literal_eval,
                                              'iv': literal_eval, 'v': literal_eval, 'vi': literal_eval,
                                              'vii': literal_eval, 'viii': literal_eval, 'ix': literal_eval,
                                              'x': literal_eval})
for col in move_df.columns:
    for name in move_df[col][0]:
        move_file('train', name, col)
