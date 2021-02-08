"""Weight init

"""
from torch.nn.init import kaiming_normal_, kaiming_uniform_
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial

from .config import Config


def _kaiming(func, module):
    """Apply Kaiming conv weight initilization

    Args:
        func (function): kaiming_normal_ or kaiming_normal_
        module (torch.nn.modules._ConvNd): The module to initilize

    """
    nonlinearity = Config.activ['name']
    mode = Config.weight_init['mode'] if 'mode' in Config.weight_init else 'fan_in'
    a = Config.activ['negative_slope'] if 'negative_slope' in Config.activ else 0
    if isinstance(module, _ConvNd):
        func(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)


def kaiming_normal(module):
    _kaiming(kaiming_normal_, module)


def kaiming_uniform(module):
    _kaiming(kaiming_uniform_, module)


def _xavier(func, module):
    gain = Config.weight_init['gain'] if 'gain' in Config.weight_init else 1
    if isinstance(module, _ConvNd):
        func(module.weight, gain=gain)


def xavier_normal(module):
    _xavier(xavier_normal_, module)


def xavier_uniform(module):
    _xavier(xavier_uniform_, module)


def init_conv_bias(module):
    if isinstance(module, _ConvNd):
        if module.bias is not None:
            constant_(module.bias, 0)


def init_norm(module):
    if isinstance(module, _BatchNorm):
        if module.bias is not None:
            constant_(module.bias, 0)
        if module.weight is not None:
            constant_(module.weight, 1)


def init_paras(model):
    """Initialize model parameters

    Args:
        model (torch.nn.Module): The model to initilize
        
    Returns:
        model (torch.nn.Module): The input model but with parameter
            initialization

    """
    weight_init = Config.weight_init['name']
    if weight_init == 'kaiming_normal':
        model.apply(kaiming_normal)
    elif weight_init == 'kaiming_uniform':
        model.apply(kaiming_uniform)
    elif weight_init == 'xavier_normal':
        model.apply(xavier_normal)
    elif weight_init == 'xavier_uniform':
        model.apply(xavier_uniform)
    model.apply(init_conv_bias)
    model.apply(init_norm)
    return model
