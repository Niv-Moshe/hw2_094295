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
def rename_data():
    for data_type in DATA_PATHS:
        for folder in os.listdir(data_type):
            for index, filename in enumerate(os.listdir(f'{data_type}/{folder}')):
                if filename == '.DS_Store':
                    os.remove(os.path.join(data_type, folder, filename))
                    print("Deleted .DS_Store file")
                    continue
                current_name = os.path.join(data_type, folder, filename)
                new_name = os.path.join(data_type, folder, f"{data_type.split('/')[-1]}_{folder}_{index}.png")
                os.rename(current_name, new_name)


# Delete images from data
def delete_files():
    delete_df = pd.read_csv("delete.csv", converters={'i': literal_eval, 'ii': literal_eval, 'iii': literal_eval,
                                                      'iv': literal_eval, 'v': literal_eval, 'vi': literal_eval,
                                                      'vii': literal_eval, 'viii': literal_eval, 'ix': literal_eval,
                                                      'x': literal_eval})
    deleted_imgs = []
    for label in delete_df.columns:
        for index in delete_df[label][0]:
            path = f'{TRAIN_PATH}/{label}/train_{label}_{index}.png'
            try:
                os.remove(path)
                print(f'Successfully deleted {path}')
            except:
                print(f'Failed to delete {path}')
            deleted_imgs.append(path)

    print(f'Deleted {len(deleted_imgs)} files')
    # print(deleted_imgs)


# Move images that are mislabeled
def move_files():
    move_df = pd.read_csv("move.csv", converters={'i': literal_eval, 'ii': literal_eval, 'iii': literal_eval,
                                                  'iv': literal_eval, 'v': literal_eval, 'vi': literal_eval,
                                                  'vii': literal_eval, 'viii': literal_eval, 'ix': literal_eval,
                                                  'x': literal_eval})
    for label in move_df.columns:  # e.g. label: 'ii'
        for name in move_df[label][0]:  # e.g. name: 'iv_33'
            old_label = name.split('_')[0]
            source = f"{TRAIN_PATH}/{old_label}/train_{name}.png"
            dest = f'{TRAIN_PATH}/{label}/train_{name}.png'
            try:
                shutil.move(source, dest)
                print(f'Successfully moved {source} to {label}')
            except:
                print(f'Failed to move {source} to {label}')


# Function to get 1 random image (from training set) and compare with transformed image
def show_single_transform(transform):
    images, filenames = [], []

    for folder in os.listdir(TRAIN_PATH):
        for image in os.listdir(TRAIN_PATH + '/' + folder):
            images.append(os.path.join(TRAIN_PATH, folder, image))
            filenames.append(f'{image}')

    random_index = random.choice(range(len(images)))
    random_filename = filenames[random_index]
    random_img = images[random_index]

    img_original = Image.open(random_img)
    img_original = ImageOps.grayscale(img_original)  # applying greyscale method

    # Execute transformation
    try:
        img_transformed = transform(img_original)
    except:
        img_transformed = transform(images=np.asarray(img_original))
    print(f'Transformation successful')

    # Display images side by side
    fig = plt.figure(figsize=(14, 8))

    # show original image
    fig.add_subplot(221)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(img_original, cmap=plt.get_cmap('gray'))
    fig.add_subplot(222)
    plt.title('Transformed Image')
    plt.axis('off')
    plt.imshow(img_transformed, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    # Pre-processing
    rename_data()
    delete_files()
    move_files()

    # Augmentation

