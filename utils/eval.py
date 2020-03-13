import torch
from torchvision import transforms
import numpy as np
from PIL import ImageDraw


# https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/extractor/DeepPixBiS.py
def predict(mask, label, threshold=0.5, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1,2,3))
        elif score_type == 'binary':
            score = torch.mean(label, axis=1)
        elif score_type == 'combined':
            score = torch.mean(mask, axis=(1,2)) + torch.mean(label, axis=1)
        else:
            raise NotImplementedError

        preds = (score > threshold).type(torch.FloatTensor)

        return preds, score
    

def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()


def add_images_tb(cfg, epoch, img_batch, preds, targets, score, writer):
    """ Do the inverse transformation
    x = z*sigma + mean
      = (z + mean/sigma) * sigma
      = (z - (-mean/sigma)) / (1/sigma),
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/6
    """
    mean = [-cfg['dataset']['mean'][i] / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['mean']))]
    sigma = [1 / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['sigma']))]
    img_transform = transforms.Compose([
        transforms.Normalize(mean, sigma),
        transforms.ToPILImage()
    ])

    ts_transform = transforms.ToTensor()

    for idx in range(img_batch.shape[0]):
        vis_img = img_transform(img_batch[idx].cpu())
        ImageDraw.Draw(vis_img).text((0,0), 'pred: {} vs gt: {}'.format(int(preds[idx]), int(targets[idx])), (255,0,255))
        ImageDraw.Draw(vis_img).text((20,20), 'score {}'.format(score[idx]), (255,0,255))
        tb_img = ts_transform(vis_img)
        writer.add_image('Prediction visualization/{}'.format(idx), tb_img, epoch)