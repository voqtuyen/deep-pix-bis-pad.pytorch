import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AverageMeter
from utils.eval import predict, calc_acc, add_images_tb


class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer)
        self.network = self.network.to(device)
        self.train_loss_metric = AverageMeter(writer=writer, name='Loss/train', length=len(self.trainloader))
        self.train_acc_metric = AverageMeter(writer=writer, name='Accuracy/train', length=len(self.trainloader))

        self.val_loss_metric = AverageMeter(writer=writer, name='Loss/val', length=len(self.testloader))
        self.val_acc_metric = AverageMeter(writer=writer, name='Accuracy/val', length=len(self.testloader))


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


    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img, mask, label) in enumerate(self.trainloader):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.network(img)
            self.optimizer.zero_grad()
            loss = self.loss(net_mask, net_label, mask, label)
            loss.backward()
            self.optimizer.step()

            # Calculate predictions
            preds = predict(net_mask, net_label, score_type=self.cfg['test']['score_type'])
            targets = predict(mask, label, score_type=self.cfg['test']['score_type'])
            acc = calc_acc(preds, targets)
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(acc)

            print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))


    def train(self):

        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            self.validate(epoch)


    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.testloader)-1)

        for i, (img, mask, label) in enumerate(self.testloader):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.network(img)
            loss = self.loss(net_mask, net_label, mask, label)

            # Calculate predictions
            preds = predict(net_mask, net_label, score_type=self.cfg['test']['score_type'])
            targets = predict(mask, label, score_type=self.cfg['test']['score_type'])
            acc = calc_acc(preds, targets)
            # Update metrics
            self.val_loss_metric.update(loss.item())
            self.val_acc_metric.update(acc)

            
            if i == seed:
                add_images_tb(self.cfg, epoch, img, preds, targets, self.writer)
