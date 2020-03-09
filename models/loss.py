from torch import nn


class PixWiseBCELoss(nn.Module):
    """ Custom loss function combining binary classification loss and pixel-wise binary loss
    Args:
        beta (float): weight factor to control weighted sum of two losses
                    beta = 0.5 in the paper implementation
    Returns:
        combined loss
    """
    def __init__(self, beta):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    
    def forward(self, net_mask, net_label, target_mask, target_label):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        loss_pixel_map = self.criterion(target_mask, net_mask)
        loss_bce = self.criterion(target_label, net_label)

        loss = self.beta * loss_bce + (1 - self.beta) * loss_pixel_map
        return loss
    
