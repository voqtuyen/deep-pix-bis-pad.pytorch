import os
import numpy as np
import torch


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, writer, name, length):
        self.reset(0)
        self.writer = writer
        self.name = name
        self.length = length


    def reset(self, epoch):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.epoch = epoch


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        self.writer.add_scalar(self.name, self.avg, self.epoch * self.length + self.count-1)

