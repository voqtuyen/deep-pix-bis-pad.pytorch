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
        self.criterion = nn.L1Loss()
        self.beta = beta

    
    def forward(self, net_mask, target_mask):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
        loss_pixel_map = self.criterion(net_mask, target_mask)

        return loss_pixel_map
