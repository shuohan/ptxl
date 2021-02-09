#!/usr/bin/env python

import os
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from pathlib import Path

from ptxl.train import SimpleTrainer, SimpleValidator
from ptxl.train import SimpleEvaluator
from ptxl.save import CheckpointSaver, ImageSaver
from ptxl.observer import Observer
from ptxl.utils import NamedData

from test_train_and_save import Dataset


def test_continue():
    os.system('rm -rf results_save_cont')

    ckpt_saver = CheckpointSaver('results_save_cont/ckpt', step=2)
    image_saver = ImageSaver('results_save_cont/images', step=2,
                             attrs=['input_cpu', 'output_cpu', 'truth_cpu'])

    loss_func = lambda x, y: torch.sum(x - y)
    loader = DataLoader(Dataset(), batch_size=4)
    net = torch.nn.Linear(1, 2, bias=False).cuda()
    optim = torch.optim.SGD(net.parameters(), lr=1, momentum=1)

    ckpt = torch.load('results_save/ckpt1/epoch-2.pt')
    trainer = SimpleTrainer(net, optim, loader, loss_func, num_epochs=3)

    trainer.register(ckpt_saver)
    trainer.register(image_saver)
    trainer.cont(ckpt)

    ref_ckpt = torch.load('results_save_te2/ckpt/epoch-3.pt')
    ckpt = torch.load('results_save_cont/ckpt/epoch-3.pt')

    print(ref_ckpt)
    print(ckpt)

    # assert ckpt.keys() == ref_ckpt.keys()
    # for key, value in ref_ckpt.items():
    #     print(type(value))
    #     assert value == ckpt[key]
    print('successful')


if __name__ == '__main__':
    test_continue()
