import os
import torch
from trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader)
        self.network = self.network.to(device)


    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])


    def save_model(self, epoch):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)


    def train(self):
        self.network.train()

        for epoch in range(self.cfg['train']['num_epochs']):
            for i, (img, mask, label) in enumerate(self.trainloader):
                img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
                net_mask, net_label = self.network(img)
                self.optimizer.zero_grad()
                loss = self.loss(net_mask, net_label, mask, label)
                loss.backward()
                self.optimizer.step()


    def validate(self):
        return