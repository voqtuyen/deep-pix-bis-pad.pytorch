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


# https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/extractor/DeepPixBiS.py
def predict(mask, label, threshold=0.5, score_type='combined'):
    if score_type == 'pixel':
        score = torch.mean(mask, axis=(1,2,3))
    elif score_type == 'binary':
        score = torch.mean(label, axis=1)
    elif score_type == 'combined':
        score = torch.mean(mask, axis=(1,2)) + torch.mean(label, axis=1)
    else:
        raise NotImplementedError

    preds = (score > threshold).type(torch.FloatTensor)

    return preds
    

def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()
