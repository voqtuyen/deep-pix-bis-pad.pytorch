import torch
from torchvision import transforms, datasets
from trainer.Trainer import Trainer
from models.loss import PixWiseBCELoss
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network


cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

loss = PixWiseBCELoss(beta=cfg['train']['loss']['beta'])

trainset = PixWiseDataset(
    root_dir=cfg['dataset']['root'],
    csv_file='',
    map_size=cfg['model']['map_size'],
    transform=None,
    smoothing=cfg['model']['smoothing']
)

testset = PixWiseDataset(
    root_dir=cfg['dataset']['root'],
    csv_file='',
    map_size=cfg['model']['map_size'],
    transform=None,
    smoothing=True
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['train']['batch_size'],
    shuffle=False,
    num_workers=2
)

trainer = Trainer(
    cfg=cfg,
    network=network,
    optimizer=optimizer,
    loss=loss,
    lr_scheduler=None,
    trainloader=trainloader,
    testloader=testloader
)

trainer.train()