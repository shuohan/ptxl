"""Utility functions and classes.

"""
from collections import namedtuple


NamedData = namedtuple('NamedData', ['name', 'data'])
"""Data with its name.

Args:
    name (str): The name of the data.
    data (numpy.ndarray): The data.
    
"""


def count_trainable_params(net):
    """Counts the trainable parameters of a network.

    Args:
        net (torch.nn.Module): The network to count.

    Returns:
        int: The number of trainable parameters.

    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
