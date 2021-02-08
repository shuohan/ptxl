#!/usr/bin/env python

import torch
from ptxl.utils import count_trainable_params


def test_count():
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3),
                              torch.nn.BatchNorm2d(8),
                              torch.nn.ReLU())

    assert count_trainable_params(net) == 72 + 8 + 16
    print('successful')

if __name__ == '__main__':
    test_count()
