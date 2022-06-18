# import os
# import numpy as np
# import torch
#
# import torchvision
# from torchvision import models, transforms, datasets
#
# import matplotlib.pyplot as plt
# import time
# import copy
# from tqdm import tqdm
#
# print("Your working directory is: ", os.getcwd())
# torch.manual_seed(0)
#
# # ======================================================
# # ======================================================
# # ======================================================
# # ======================================================
#
# # You are not allowed to change anything in this file.
# # This file is meant only for training and saving the model.
# # You may use it for basic inspection of the model performance.
#
# # ======================================================
# # ======================================================
# # ======================================================
# # ======================================================
#
# # Training hyperparameters
# BATCH_SIZE = 16
# NUM_EPOCHS = 100
# LR = 0.001
#
# # Paths to your train and val directories
# train_dir = os.path.join("data", "train")
# val_dir = os.path.join("data", "val")
#
#
# def imshow(inp, title=None):
#     """Imshow for Tensors."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     plt.figure(figsize=(15, 15))
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#
#
# # Resize the samples and transform them into tensors
# data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
#
# # Create a pytorch dataset from a directory of images
# train_dataset = datasets.ImageFolder(train_dir, data_transforms)
# val_dataset = datasets.ImageFolder(val_dir, data_transforms)
#
# class_names = train_dataset.classes
# print("The classes are: ", class_names)
#
# # Dataloaders initialization
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
#
# for batch in train_dataloader:
#     inputs, targets = batch
#     for img in inputs:
#         image  = img.cpu().numpy()
#         # # transpose image to fit plt input
#         image = image.T
#         # # normalise image
#         data_min = np.min(image, axis=(1,2), keepdims=True)
#         data_max = np.max(image, axis=(1,2), keepdims=True)
#         scaled_data = (image - data_min) / (data_max - data_min)
#         # show image
#         plt.imshow(image)
#         plt.show()
#
#

import numpy as np
from sklearn.model_selection import train_test_split

hi = [f'hi_{i}' for i in range(100)]
train, val = train_test_split(hi, shuffle=True, train_size=80)
print(val)
print(len(val))
