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
from collections import Counter, OrderedDict
from torchvision import models, transforms, datasets
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from transformations import transform_no_flip, transform_horizontal, transform_vertical, transform_horizontal_vertical
# Set seed
ia.seed(1)

# New folder for clean data
CLEAN_DATA_PATH = 'data_clean'
TRAIN_PATH = f'{CLEAN_DATA_PATH}/train'
VALID_PATH = f'{CLEAN_DATA_PATH}/val'
DATA_PATHS = [TRAIN_PATH, VALID_PATH]

# Delete data_clean folder is exists
if os.path.exists(CLEAN_DATA_PATH):
    shutil.rmtree(CLEAN_DATA_PATH)
    print("Deleted data_clean folder")

# Create new directory for cleaned data
shutil.copytree('data', CLEAN_DATA_PATH)
print('Created data_clean folder')

# Delete data_augmented folder is exists
if os.path.exists('data_augmented'):
    shutil.rmtree('data_augmented')
    print("Deleted data_augmented folder")
print()


# Exploring the data
def exploration():
    print('Exploring the data:')
    # number of images
    train_len = len([os.path.join(path, filename) for path, _, filenames in os.walk(TRAIN_PATH)
                     for filename in filenames])
    print(f'Train length: {train_len}')
    valid_len = len([os.path.join(path, filename) for path, _, filenames in os.walk(VALID_PATH)
                     for filename in filenames])
    print(f'Valid length: {valid_len}')

    # images dimensions
    all_dimensions = []
    for folder in os.listdir(TRAIN_PATH):
        for filename in os.listdir(f'{TRAIN_PATH}/{folder}'):
            if filename == '.DS_Store':
                print('Skipped .DS_Store file')
                continue
            path = os.path.join(TRAIN_PATH, folder, filename)
            image = ImageOps.grayscale(Image.open(path))
            all_dimensions.append(np.asarray(image).shape)
    print(f'Dimensions count: {Counter(all_dimensions).most_common()}')
    print()


# Rename images files in train and val sets
def rename_data():
    print('Renaming data:')
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
    print()


# Delete images from data
def delete_files():
    print('Deleting invalid samples:')
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
                print(f'Deleted {path}')
            except:
                print(f'Error in deleting {path}')
            deleted_imgs.append(path)

    print(f'Deleted {len(deleted_imgs)} files')
    # print(f'All deleted images: {deleted_imgs}')
    print()


# Move images that are mislabeled
def move_files():
    print('Moving wrong labeled samples to correct labels:')
    move_df = pd.read_csv("move.csv", converters={'i': literal_eval, 'ii': literal_eval, 'iii': literal_eval,
                                                  'iv': literal_eval, 'v': literal_eval, 'vi': literal_eval,
                                                  'vii': literal_eval, 'viii': literal_eval, 'ix': literal_eval,
                                                  'x': literal_eval})
    count_moved = 0
    for label in move_df.columns:  # e.g. label: 'ii'
        for name in move_df[label][0]:  # e.g. name: 'iv_33'
            old_label = name.split('_')[0]
            source = f"{TRAIN_PATH}/{old_label}/train_{name}.png"
            destination = f'{TRAIN_PATH}/{label}/train_{name}.png'
            try:
                shutil.move(source, destination)
                print(f'Moved {source} to {label}')
                count_moved += 1
            except:
                print(f'Error in moving {source} to {label}')
    print(f'Moved {count_moved} files')
    print()


# Merging train and val to same folder, augmenting and new train val split
def augmentation_and_split(label, transformation, max_size=1000, train_size=800):
    print(f'Generating images for label={label}:')
    # creating a temporary folder for all (train+val) the clean images of label
    temp_folder = f'data_augmented/temp_{label}'
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    # merging (copying) the clean images to a temporary folder
    for path in [TRAIN_PATH, VALID_PATH]:
        label_folder = f'{path}/{label}'
        for image in os.listdir(label_folder):
            shutil.copy(f'{label_folder}/{image}', temp_folder)

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
    indexes = list(range(len(clean_images)))
    for i in range(images_to_generate_count):
        index = random.choice(indexes)
        image = ImageOps.grayscale(Image.open(clean_images[index]))
        # transforming
        image_transformed = transformation(images=np.asarray(image))
        image_transformed = Image.fromarray(image_transformed)
        image_transformed.save(f'{temp_folder}/random{i}_{clean_images[index].split("/")[-1]}', 'PNG')

    # splitting to train and val
    print(f'Total images {len(os.listdir(temp_folder))}')
    train, val = train_test_split(os.listdir(temp_folder), shuffle=True, train_size=train_size)
    # copying to train and val before deleting temp_folder
    for train_file_path in train:
        shutil.copy(os.path.join(temp_folder, train_file_path), augmented_label_folder_train)
    for val_file_path in val:
        shutil.copy(os.path.join(temp_folder, val_file_path), augmented_label_folder_val)

    # deleting temp_folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    print()


def make_confusion_matrix():
    val_dir = os.path.join("data", "val")
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms)
    len_val = len(val_dataset)
    class_names = val_dataset.classes
    NUM_CLASSES = len(class_names)
    print("The classes are: ", class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len_val, shuffle=True)
    # Load
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(device=device)
    model_ft.load_state_dict(torch.load("trained_model.pt"))
    model_ft.eval()

    # evaluating on validation the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(val_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # Build confusion matrix
        cf_matrix = confusion_matrix(labels.data.detach().cpu(), preds.detach().cpu())
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in class_names],
                             columns=[i for i in class_names])

        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('confusion_matrix.png')
    epoch_loss = running_loss / len_val
    epoch_acc = running_corrects.double() / len_val
    print(f'Val acc: {epoch_acc}, Val loss: {epoch_loss}')


def main():
    # Exploration
    exploration()
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


if __name__ == "__main__":
    main()
    # make_confusion_matrix()
