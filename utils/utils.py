import yaml
from torch import optim
from models.densenet_161 import DeepPixBis


def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg


def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.paramters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer


def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None

    if cfg['model']['name'] == 'densenet_161':
        network = DeepPixBis(pretrained=cfg['model']['pretrained'])
    else:
        raise NotImplementedError

    return network