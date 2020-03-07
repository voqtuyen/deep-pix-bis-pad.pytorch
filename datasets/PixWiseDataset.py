import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import io
from torchvision import transforms


class PixWiseDataset(Dataset):
    """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way

    Args:
        root_dir (string): Root directory path
        csv_file (string): csv file to dataset annotation
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """

    def __init__(self, root_dir, csv_file, map_size, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.map_size = map_size
        self.transform = transform
        
        if smoothing:
            self.label_weight = 0.99
        else:
            self.label_weight = 1.0


    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_name)
        img = io.imread(img_name)

        label = self.data.iloc[index, 1]

        if label == 1:
            mask = np.ones((self.map_size, self.map_size), dtype=np.float) * self.label_weight
        else:
            mask = np.ones((self.map_size, self.map_size)) * (1.0 - self.label_weight)

        if self.transform:
            img = self.transform(img)

        return img, label, mask


    def __len__(self):
        return len(self.data)
