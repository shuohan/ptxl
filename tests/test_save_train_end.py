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


def test_save_train_end():
    os.system('rm -rf results_save_te2')
    os.system('rm -rf results_save_te0')

    ckpt_saver = CheckpointSaver('results_save_te2/ckpt', step=2) 
    image_saver = ImageSaver('results_save_te2/images', step=2,
                             attrs=['input_cpu', 'output_cpu', 'truth_cpu'])

    net = torch.nn.Linear(1, 2, bias=False).cuda()
    torch.nn.init.zeros_(net.weight)
    optim = torch.optim.SGD(net.parameters(), lr=1, momentum=1)
    loader = DataLoader(Dataset(), batch_size=4)
    loss_func = lambda x, y: torch.sum(x - y)
    trainer = SimpleTrainer(net, optim, loader, loss_func, num_epochs=3)
    assert trainer.num_batches == 3
    assert trainer.num_epochs == 3
    assert trainer.batch_size == 4

    trainer.register(ckpt_saver)
    trainer.register(image_saver)
    trainer.train()

    ckpt = torch.load('results_save_te2/ckpt/epoch-3.pt')
    print(ckpt)

    ckpt_saver = CheckpointSaver('results_save_te0/ckpt', step=0) 
    image_saver = ImageSaver('results_save_te0/images', step=0,
                             attrs=['input_cpu', 'output_cpu', 'truth_cpu'])

    net = torch.nn.Linear(1, 2, bias=False).cuda()
    torch.nn.init.zeros_(net.weight)
    optim = torch.optim.SGD(net.parameters(), lr=1, momentum=1)
    trainer = SimpleTrainer(net, optim, loader, loss_func, num_epochs=2)

    trainer.register(ckpt_saver)
    trainer.register(image_saver)
    trainer.train()

    print('successful')


if __name__ == '__main__':
    test_save_train_end()
