from trainer.base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, trainloader, testloader):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, trainloader, testloader)


    def load_model(self):
        return


    def save_model(self):
        return


    def train(self):
        return


    def validate(self):
        return