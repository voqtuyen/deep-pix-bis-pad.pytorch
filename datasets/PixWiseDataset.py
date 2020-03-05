import torch
from torch.utils.data import Dataset


class PixWiseDataset(Dataset):
    """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way:
            root/genuine/xxx.ext
            root/genuine/xxy.ext
            root/genuine/xxz.ext

            root/fake/123.ext
            root/fake/nsdf3.ext
            root/fake/asd932_.ext

    Args:
        root (string): Root directory path
        transform: A function/transform that takes in a sample and returns a transformed version
    """
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform


    def __getitem__(self, idx):
        return


    def __len__(self):
        return 1