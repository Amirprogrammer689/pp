import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random


class ImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        
        label = os.path.basename(img_path).split('.')[0]
        label = 1 if label == 'polar bear' else 0
        
        return image, label


def load_images(df, path, indices, img_type):
    for i in indices:
        image_path = os.path.abspath(os.path.join(*df.full_path[i].split("\\")))
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(path, f'{df.img_class[i]}.{i}.{img_type}.jpg'), image)


device = 'cpu'

train_path = os.path.abspath('train_list')
test_path = os.path.abspath('test_list')
val_path = os.path.abspath('val_list')

# ... (code for reading and processing the CSV file)
df = pd.read_csv('annotations_1.csv', sep = ',', header=None)

df = df.drop(df.index[0])
df.drop(0, axis=1, inplace=True)
df.rename(columns={1: 'full_path', 2: 'img_class'}, inplace=True)
df.reset_index(inplace=True)
print(df.sample(10))

# Create directories if they don't exist
for path in [train_path, test_path, val_path]:
    if not os.path.isdir(path):
        os.mkdir(path)

# Load images for training, testing, and validation
for i in range(800):
    load_images(df, train_path, i, 'train')
for i in range(1000, 1800):
    load_images(df, train_path, i, 'train')

for i in range(800, 900):
    load_images(df, test_path, i, 'test')
for i in range(1800, 1900):
    load_images(df, test_path, i, 'test')

for i in range(900, 1000):
    load_images(df, val_path, i, 'val')
for i in range(1900, 1999):
    load_images(df, val_path, i, 'val')

# Split train_list into train and validation sets
train_list = glob.glob(os.path.join(train_path,'*.jpg'))
train_list, val_list = train_test_split(train_list, test_size=0.1)

# Display random images from the training set
random_idx = np.random.choice(len(train_list), size=10, replace=False)
fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    img = cv2.imread(train_list[random_idx[i]])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Define transformations for training, validation, and testing data
common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_data = ImageDataset(train_list, transform=common_transforms)
val_data = ImageDataset(val_list, transform=common_transforms)
test_data = ImageDataset(glob.glob(os.path.join(test_path,'*.jpg')), transform=common_transforms)

print(train_data[0])
print(val_data[0])
