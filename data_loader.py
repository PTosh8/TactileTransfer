import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools
from matplotlib import image
import glob as glob
from PIL import Image

from silx.math import colormap
import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary

import cv2

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Using the GPU!")
else:
  print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")


class TactileTransferBubbles(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: the directory of the dataset
            split: "train" or "val"
            transform: pytorch transformations.
        """

        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, '*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.load(self.files[idx])['bubble_imprint'][0]
        img = img.permute(1,2,0).numpy()
        if self.transform:
            img = self.transform(img)
        return img

class TactileTransfer(Dataset):
    def __init__(self, root_dir, bubbles_transform=None, gelslim_transform=None):
        """
        Args:
            root_dir: the directory of the dataset
            split: "train" or "val"
            transform: pytorch transformations.
        """

        self.bubbles_transform = bubbles_transform
        self.gelslim_transform = gelslim_transform
        self.bubbles_files = glob.glob(os.path.join(root_dir, 'bubbles_cone', '*.pt'))
        self.gelslim_files = glob.glob(os.path.join(root_dir, 'gelslim_cone', '*.pt'))

    def __len__(self):
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        bubbles_img = torch.load(self.bubbles_files[idx])['bubble_imprint']
        bubbles_img = bubbles_img[0,:,:,:].permute(1,2,0).numpy()
        if self.bubbles_transform:
            bubbles_img = self.bubbles_transform(bubbles_img)

        gelslim_img = torch.load(self.gelslim_files[idx])['gelslim'][0]
        gelslim_img = gelslim_img.permute(1,2,0).numpy()
        if self.gelslim_transform:
            gelslim_img = self.gelslim_transform(gelslim_img)

        return bubbles_img, gelslim_img


class MyTransform:

    def __init__(self):
        pass
    
    def __call__(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

def main():
    bubbles_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize(mean=(0.0028362185694277287), std=(0.004623861517757177)),
        MyTransform()
    ])

    gelslim_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128))
    ])

    dataset = TactileTransfer('data', bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform)

    train_set, test_set, val_set = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator())
    training_indices = train_set.indices
    testing_indices = test_set.indices
    val_indices = val_set.indices

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


    print('Number of bubbles images {}, number of gelslim images {}'.format(len(dataset.bubbles_files), len(dataset.gelslim_files)))

    idx = 0
    for batch in train_loader:
        filename = f'data_{training_indices[idx]}.jpg'

        bubbles_image = batch[0]
        bubbles_np_image = (bubbles_image.squeeze(0).permute(1,2,0).numpy())
        colorized = cv2.applyColorMap((bubbles_np_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('processed_data', 'processed_bubbles_cone', 'train', filename), colorized)

        gelslim_image = batch[1]
        gelslim_np_image = (np.rint(gelslim_image.squeeze(0).permute(1,2,0).numpy()).astype(np.uint8))
        cv2.imwrite(os.path.join('processed_data', 'processed_gelslim_cone', 'train', filename), gelslim_np_image)
        
        idx += 1
    
    idx = 0
    for batch in test_loader:
        filename = f'data_{testing_indices[idx]}.jpg'

        bubbles_image = batch[0]
        bubbles_np_image = (bubbles_image.squeeze(0).permute(1,2,0).numpy())
        colorized = cv2.applyColorMap((bubbles_np_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('processed_data', 'processed_bubbles_cone', 'test', filename), colorized)

        gelslim_image = batch[1]
        gelslim_np_image = (np.rint(gelslim_image.squeeze(0).permute(1,2,0).numpy()).astype(np.uint8))
        cv2.imwrite(os.path.join('processed_data', 'processed_gelslim_cone', 'test', filename), gelslim_np_image)
        
        idx += 1

    idx = 0
    for batch in val_loader:
        filename = f'data_{val_indices[idx]}.jpg'

        bubbles_image = batch[0]
        bubbles_np_image = (bubbles_image.squeeze(0).permute(1,2,0).numpy())
        colorized = cv2.applyColorMap((bubbles_np_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('processed_data', 'processed_bubbles_cone', 'val', filename), colorized)

        gelslim_image = batch[1]
        gelslim_np_image = (np.rint(gelslim_image.squeeze(0).permute(1,2,0).numpy()).astype(np.uint8))
        cv2.imwrite(os.path.join('processed_data', 'processed_gelslim_cone', 'val', filename), gelslim_np_image)
        
        idx += 1

    # mean = 0.0
    # for images in gelslim_loader:
    #     batch_samples = images.size(0) 
    #     images = images.view(batch_samples, -1)
    #     mean += torch.mean(images) * batch_samples
    # mean = mean / len(gelslim_loader.dataset)

    # var = 0.0
    # for images in gelslim_loader:
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, -1)
    #     var += torch.std(images) ** 2
    # print(f'Mean = {torch.mean(mean)}, std = {torch.sqrt(var)}')

if __name__=="__main__":
    main()
