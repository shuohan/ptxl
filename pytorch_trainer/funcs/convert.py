# -*- coding: utf-8 -*-

"""Functions to convert data and models."""

import torch


def convert_th_to_np(data):
    """Converts :class:`torch.Tensor` to :class:`numpy.ndarray'.

    Args:
        data (torch.Tensor): The tensor to convert.

    Returns:
        numpy.ndarray: The converted.

    """
    return data.detach().cpu().numpy()


def _eval(instance, func, Type, *args, **kwargs):
    """Calls the function of the instance.

    Args:
        data (tuple or list or dict or Type): The instance to call the function
            from.

    Returns:
        tuple or list or dict or Type: The results of the function called.

    """
    if type(instance) is tuple:
        return tuple(_eval(i, func, Type, *args, **kwargs) for i in instance)
    elif type(instance) is list:
        return list(_eval(i, func, Type, *args, **kwargs) for i in instance)
    elif type(instance) is dict:
        return {k: _eval(v, func, Type, *args, **kwargs)
                for k, v in instance.items()}
    elif isinstance(instance, Type):
        return getattr(instance, func)(*args, **kwargs)


def transfer_data_to_cuda(data, *args, **kwargs):
    """Transfers data into GPU.

    Args:
        data (tuple or list or dict or torch.Tensor): The data to transfer.

    Returns:
        tuple or list or dict or torch.Tensor: The transferred data.

    """
    return _eval(data, 'cuda', torch.Tensor, *args, **kwargs)


def transfer_data_to_cpu(data, *args, **kwargs):
    """Transfers data into CPU.

    Args:
        data (tuple or list or dict or torch.Tensor): The data to transfer.

    Returns:
        tuple or list or dict or torch.Tensor: The transferred data.

    """
    return _eval(data, 'cpu', torch.Tensor, *args, **kwargs)


def transfer_models_to_cuda(models, *args, **kwargs):
    """Transfers models into GPU.
    
    Args:
        models (tuple or list or dict or torch.nn.Module): The models to
            transfer.

    Returns:
        tuple or list or dict or torch.nn.Module: The transferred models.
    
    """
    return _eval(models, 'cuda', torch.nn.Module, *args, **kwargs)


def transfer_models_to_cpu(models, *args, **kwargs):
    """Transfers models into CPU.
    
    Args:
        models (tuple or list or dict or torch.nn.Module): The models to
            transfer.

    Returns:
        tuple or list or dict or torch.nn.Module: The transferred models.
    
    """
    return _eval(models, 'cpu', torch.nn.Module, *args, **kwargs)


def set_models_to_train(models, *args, **kwargs):
    """Sets models into train mode.
    
    Args:
        models (tuple or list or dict or torch.nn.Module): The models to change.

    Returns:
        tuple or list or dict or torch.nn.Module: The transferred models.
    
    """
    return _eval(models, 'train', torch.nn.Module, *args, **kwargs)


def set_models_to_eval(models, *args, **kwargs):
    """Sets models into eval mode.
    
    Args:
        models (tuple or list or dict or torch.nn.Module): The models to change.

    Returns:
        tuple or list or dict or torch.nn.Module: The transferred models.
    
    """
    return _eval(models, 'eval', torch.nn.Module, *args, **kwargs)
