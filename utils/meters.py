import os
import numpy as np
import torch


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


# https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/extractor/DeepPixBiS.py
def calc_acc(mask, label, threshold=0.5, score_type='pixel'):
    if score_type == 'pixel':
        score = np.mean(mask, axis=(1,2))
    elif score_type == 'binary':
        score = label 
    elif score_type == 'combined':
        score = np.mean(mask, axis=(1,2)) + label
    else:
        raise NotImplementedError

    score = score > threshold

    return score
    
