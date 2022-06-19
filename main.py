import imgaug as ia
import os
from pathlib import Path
from glob import glob
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ast import literal_eval
from sklearn.model_selection import train_test_split
from transformations import transform_no_flip, transform_horizontal, transform_vertical, transform_horizontal_vertical
# Set seed
ia.seed(1)

# New folder for clean data
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

# Remove data_augmented folder is exists
if os.path.exists('data_augmented'):
    shutil.rmtree('data_augmented')
    print("Removed data_augmented folder")


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
            destination = f'{TRAIN_PATH}/{label}/train_{name}.png'
            try:
                shutil.move(source, destination)
                print(f'Successfully moved {source} to {label}')
            except:
                print(f'Failed to move {source} to {label}')


# Merging train and val to same folder, augmenting and new train val split
def augmentation_and_split(label, transformation, max_size=1000, train_size=800):
    # creating a temporary folder for all (train+val) the clean images of label
    temp_folder = f'data_augmented/temp_{label}'
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    # merging (copying) the clean images to a temporary folder
    for path in [TRAIN_PATH, VALID_PATH]:
        label_folder = f'{path}/{label}'
        for image in os.listdir(label_folder):
            shutil.copy(f'{label_folder}/{image}', temp_folder)
    print(f'Temp folder created: {temp_folder}')

    # train and val paths of augmented images for label
    augmented_label_folder_train = f'data_augmented/train/{label}'
    augmented_label_folder_val = f'data_augmented/val/{label}'
    # creating the folders
    Path(augmented_label_folder_train).mkdir(parents=True, exist_ok=True)
    Path(augmented_label_folder_val).mkdir(parents=True, exist_ok=True)

    # number of images to generate to reach total_size=1000 images
    images_to_generate_count = max_size - len(os.listdir(temp_folder))
    # all clean images paths from temp folder
    clean_images = glob(os.path.join(temp_folder, "*.png"))
    print(f'Images to generate {images_to_generate_count}')
    for i in range(images_to_generate_count):
        index = random.choice(range(len(clean_images)))
        image = Image.open(clean_images[index])
        image = ImageOps.grayscale(image)
        # transforming
        image_transformed = transformation(images=np.asarray(image))
        image_transformed = Image.fromarray(image_transformed)
        image_transformed.save(f'{temp_folder}/random{i}_{clean_images[index].split("/")[-1]}', 'PNG')

    # splitting to train and val
    print(f'Total images: {len(os.listdir(temp_folder))}')
    train, val = train_test_split(os.listdir(temp_folder), shuffle=True, train_size=train_size)
    # copying to train and val before deleting temp_folder
    for train_file_path in train:
        shutil.copy(os.path.join(temp_folder, train_file_path), augmented_label_folder_train)
    for val_file_path in val:
        shutil.copy(os.path.join(temp_folder, val_file_path), augmented_label_folder_val)

    # deleting temp_folder
    shutil.rmtree(temp_folder, ignore_errors=True)


if __name__ == "__main__":
    # Pre-processing
    rename_data()
    delete_files()
    move_files()

    # Augmentation
    labels = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    labels_no_flip = ['iv', 'vi', 'vii', 'viii']
    labels_horizontal = ['v']
    labels_vertical = ['ix']
    labels_horizontal_vertical = ['i', 'ii', 'iii', 'x']
    # executing transformations
    for label in labels:
        if label in labels_no_flip:
            augmentation_and_split(label=label, transformation=transform_no_flip)

        if label in labels_horizontal:
            augmentation_and_split(label=label, transformation=transform_horizontal)

        if label in labels_vertical:
            augmentation_and_split(label=label, transformation=transform_vertical)

        if label in labels_horizontal_vertical:
            augmentation_and_split(label=label, transformation=transform_horizontal_vertical)

