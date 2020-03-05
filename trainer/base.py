class BaseTrainer():
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, trainloader, testloader):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.trainloader = trainloader
        self.testloader = testloader

    
    def load_model(self):
        raise NotImplementedError


    def save_model(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def validate(self):
        raise NotImplementedError