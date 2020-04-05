import os
import torch
from torchvision import transforms, datasets
from trainer.Trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from models.loss import PixWiseBCELoss
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cfg = read_cfg(cfg_file='config/patch-based_lr1e-3.yaml')

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

loss = PixWiseBCELoss(beta=cfg['train']['loss']['beta'])

writer = SummaryWriter(cfg['log_dir'])

dump_input = torch.randn(1,3,256,256)

writer.add_graph(network, (dump_input, ))

# Without Resize transform, images are of different sizes and it causes an error
train_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rotation']),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

test_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
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
    num_workers=0
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['test']['batch_size'],
    shuffle=True,
    num_workers=0
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
