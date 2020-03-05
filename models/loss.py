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

    
    def forward(self, output, label):
        loss = 0
        return loss
    
