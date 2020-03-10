import os
import cv2
import torch
from torchvision import transforms, datasets
from models.loss import PixWiseBCELoss
from models.densenet_161 import DeepPixBis
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device


cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

network = build_network(cfg)

checkpoint = torch.load(os.path.join(cfg['output_dir'], '{}_{}.pth'.format(cfg['model']['base'], cfg['dataset']['name'])))

network.load_state_dict(checkpoint['state_dict'])

img = cv2.imread()