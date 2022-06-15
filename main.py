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
# Set seed
ia.seed(1)

TRAIN_DATA = 'data/train'
VALID_DATA = 'data/val'
print(f'Train data folder: {os.listdir(TRAIN_DATA)}')
print(f'Validation data folder: {os.listdir(VALID_DATA)}')

# Print random images
# images, labels = [], []
#
# for folder in os.listdir(TRAIN_DATA):
#     for image in os.listdir(TRAIN_DATA + '/' + folder):
#         images.append(os.path.join(TRAIN_DATA, folder, image))
#         labels.append(folder)
#
# plt.figure(1, figsize=(15, 9))
# plt.axis('off')
# n = 0
#
# for i in range(36):
#     n += 1
#     random_index = random.choice(range(len(images)))
#     random_label = labels[random_index]
#     random_img = images[random_index]
#     img = cv2.imread(random_img, cv2.IMREAD_GRAYSCALE)
#     plt.subplot(6, 6, n)
#     plt.axis('off')
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#     plt.title(random_label)
#
# plt.show()

# Create new directory to store cleaned images (copy raw data)
CLEAN_DATA_FOLDER = 'data_clean'
TRAIN_DATA = f'{CLEAN_DATA_FOLDER}/train'
VALID_DATA = f'{CLEAN_DATA_FOLDER}/val'

try:
    shutil.copytree('data', CLEAN_DATA_FOLDER, dirs_exist_ok=False)
    print('Created new folder')
except:
    print('Folder already exists')

# Rename files in train and val sets (in new clean folder) for easier tracking
# data_types = [TRAIN_DATA, VALID_DATA]
#
# for data_type in data_types:
#     for folder in os.listdir(data_type):
#         for index, file in enumerate(os.listdir(data_type + '/' + folder)):
#             data_type_name = data_type.split('/')[-1]
#             os.rename(os.path.join(data_type, folder, file), os.path.join(data_type, folder,
#                                                                           ''.join(
#                                                                               [str(data_type_name), '_', str(folder),
#                                                                                '_', str(index), '.png'])))


# Setup function to display random images based on a label (for manual review) - Includes both train and valid
def show_images(label, num):
    images, labels, filenames = [], [], []

    # Add train data images
    data_folders = [TRAIN_DATA, VALID_DATA]

    for data_folder in data_folders:
        for folder in os.listdir(data_folder):
            for image in os.listdir(data_folder + '/' + label):
                images.append(os.path.join(data_folder, label, image))
                filenames.append(f'{image}')

    plt.figure(1, figsize=(18, 11))
    plt.axis('off')

    n = 0
    for i in range(num):
        n += 1
        random_index = random.choice(range(len(images)))
        # random_index = int(filenames.index('train_iii_105.png')) # can't read the file
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


show_images('iii', 16)

# Setup function to add file into list for deleting later
deleted_imgs = []


def delete_file(data_type, img_name):
    global deleted_imgs
    label = img_name.split('_')[0]
    file_path = f'{CLEAN_DATA_FOLDER}/{data_type}/{label}/{img_name}.png'

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
    source = f'{CLEAN_DATA_FOLDER}/{data_type}/{old_label}/{img_name}.png'
    dest = f'{CLEAN_DATA_FOLDER}/{data_type}/{new_label}/{img_name}.png'

    try:
        shutil.move(source, dest)
        print(f'Successfully moved {source}')
    except:
        pass


# Files to move
# move_file('train', 'i_70', 'ii')
# move_file('train', 'i_98', 'ii')
# move_file('train', 'i_123', 'iii')
#
# # Add images to list for deletion
# delete_file('train', 'i_7')
# delete_file('train', 'i_75')
# delete_file('train', 'i_76')  # Unclear, few pixels
# delete_file('train', 'i_122')
