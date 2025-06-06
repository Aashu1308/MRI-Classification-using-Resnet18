import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from numpy.random import seed
import random
# import numpy as np
import shutil
# from glob import glob
# from textwrap import wrap

seed(42)
torch.manual_seed(42)

def create_validation_split(tr_path, val_path, split_ratio=0.1):
    os.makedirs(val_path, exist_ok=True)
    for cls in os.listdir(tr_path):
        cls_path = os.path.join(tr_path, cls)
        if not os.path.isdir(cls_path):
            continue
        os.makedirs(os.path.join(val_path, cls), exist_ok=True)
        images = os.listdir(cls_path)
        random.shuffle(images)
        val_size = int(len(images) * split_ratio)
        for img in images[:val_size]:
            shutil.move(os.path.join(cls_path, img), os.path.join(val_path, cls, img))

def train_df(tr_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(tr_path, label, image))
            for label in os.listdir(tr_path)
            if os.path.isdir(os.path.join(tr_path, label))
            for image in os.listdir(os.path.join(tr_path, label))
        ]
    )

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(ts_path, label, image))
            for label in os.listdir(ts_path)
            if os.path.isdir(os.path.join(ts_path, label))
            for image in os.listdir(os.path.join(ts_path, label))
        ]
    )

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df


def visualise(tr_df,folder_name):
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(data=tr_df, y=tr_df['Class'])
    plt.title(f'Count of images in each class of {folder_name}', fontsize=20)
    ax.bar_label(ax.containers[0])
    plt.show()


# class CustomDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.dataframe = dataframe
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx, 0]
#         image = Image.open(img_path).convert('RGB')
#         label = self.dataframe.iloc[idx, 1]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

def list_images(tr_path, classes):
    original = []
    augmented = []
    for cls in classes:
        cls_path = os.path.join(tr_path, cls)
        for fname in os.listdir(cls_path):
            if fname.startswith('aug_'):
                augmented.append(os.path.join(cls_path, fname))
            else:
                original.append(os.path.join(cls_path, fname))
    return original, augmented



# def view_with_class(train_loader, class_dict):
#     images, labels = next(iter(train_loader))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     images = images.permute(0, 2, 3, 1).numpy()
#     images = images * std + mean
#     images = np.clip(images, 0, 1)
#     num_images_to_display = min(len(images), 16)
#     plt.figure(figsize=(20, 20))
#     for i, (image, label) in enumerate(zip(images[:num_images_to_display], labels[:num_images_to_display])):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(image)
#         class_name = label
#         plt.title(class_name, color='k', fontsize=15)
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()