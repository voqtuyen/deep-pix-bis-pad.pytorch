import os
import torch
from torchvision import transforms, datasets
from trainer.Trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from models.loss import PixWiseBCELoss
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device
from utils.transforms import RandomGammaCorrection, RandomBrightness


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

cfg = read_cfg(cfg_file='config/patch-based_v2_lr1e-3.yaml')

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

loss = PixWiseBCELoss(beta=cfg['train']['loss']['beta'])

writer = SummaryWriter(cfg['log_dir'])

dump_input = torch.randn(1,3,320,320)

# writer.add_graph(network, dump_input, dump_input)

# Without Resize transform, images are of different sizes and it causes an error
train_transform = transforms.Compose([
    RandomGammaCorrection(),
    transforms.RandomResizedCrop(320),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rotation']),
    transforms.RandomHorizontalFlip()
])

test_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size'])
])

trainset = PixWiseDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['train_set'],
    map_size=cfg['model']['map_size'],
    transform=train_transform,
    smoothing=cfg['model']['smoothing']
)

testset = PixWiseDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['test_set'],
    map_size=cfg['model']['map_size'],
    transform=test_transform,
    smoothing=cfg['model']['smoothing']
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['test']['batch_size'],
    shuffle=True,
    num_workers=2
)

trainer = Trainer(
    cfg=cfg,
    network=network,
    optimizer=optimizer,
    loss=loss,
    lr_scheduler=None,
    device=device,
    trainloader=trainloader,
    testloader=testloader,
    writer=writer
)

trainer.train()
writer.close()
