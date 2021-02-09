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


class Dataset:
    def __len__(self):
        return 12
    def __getitem__(self, ind):
        x = NamedData(name='x_%d' % ind, data=torch.Tensor([ind]))
        y = NamedData(name='y_%d' % ind, data=torch.Tensor([ind + 1]))
        return x, y


class Check(Observer):
    def update_on_batch_end(self):
        if self.subject.batch_ind == 1:
            assert self.subject.net.weight.grad.tolist() == [[6], [6]]
        elif self.subject.batch_ind == 2:
            assert self.subject.net.weight.grad.tolist() == [[22], [22]]
        elif self.subject.batch_ind == 3:
            assert self.subject.net.weight.grad.tolist() == [[38], [38]]


def test_train_and_save():
    os.system('rm -rf results_save')

    tmp = {'first': 1, 'second': [1, 2, 3]}
    ckpt_saver1 = CheckpointSaver('results_save/ckpt1', step=1, **tmp)
    ckpt_saver2 = CheckpointSaver('results_save/ckpt2', step=2, save_init=True)
    train_saver = ImageSaver('results_save/train', step=2,
                             attrs=['input_cpu', 'output_cpu', 'truth_cpu'])
    valid_saver = ImageSaver('results_save/valid', step=2,
                             attrs=['input_cpu', 'output_cpu', 'truth_cpu'])

    net = torch.nn.Linear(1, 2, bias=False).cuda()
    torch.nn.init.zeros_(net.weight)
    optim = torch.optim.SGD(net.parameters(), lr=1, momentum=1)
    loader = DataLoader(Dataset(), batch_size=4)
    loss_func = lambda x, y: torch.sum(x - y)
    trainer = SimpleTrainer(net, optim, loader, loss_func, num_epochs=2)
    assert trainer.num_batches == 3
    assert trainer.num_epochs == 2
    assert trainer.batch_size == 4

    loader = DataLoader(Dataset(), batch_size=3)
    validator = SimpleValidator(loader, step=1)
    validator.register(valid_saver)

    evaluator = SimpleEvaluator({'sum': lambda x, y: torch.sum(x + y)})

    trainer.register(validator)
    trainer.register(evaluator)
    trainer.register(ckpt_saver1)
    trainer.register(ckpt_saver2)
    trainer.register(train_saver)
    trainer.register(Check())
    trainer.train()

    assert evaluator.sum == -20132
    assert evaluator.batch_size == 4

    assert validator.batch_size == 3
    assert validator.num_batches == 4
    assert validator.num_epochs == 2
    assert validator.loss == -23946

    ckpt = torch.load('results_save/ckpt1/epoch-2.pt')
    assert ckpt['first'] == 1
    assert ckpt['second'] == [1, 2, 3]
    assert ckpt['first'] == 1
    assert torch.allclose(ckpt['model_state_dict']['weight'],
                          torch.FloatTensor([[-398], [-398]]).cuda())
    assert ckpt['optim_state_dict']['state'][0]['momentum_buffer'].tolist() \
        == [[132], [132]]
    assert ckpt['epoch'] == 2

    ckpt = torch.load('results_save/ckpt2/epoch-0.pt')
    assert torch.allclose(ckpt['model_state_dict']['weight'],
                         torch.FloatTensor([[0], [0]]).cuda())
    assert ckpt['optim_state_dict']['state'] == {}
    assert ckpt['epoch'] == 0

    assert Path('results_save/ckpt1/epoch-1.pt').is_file()
    assert Path('results_save/ckpt1/epoch-2.pt').is_file()
    assert Path('results_save/ckpt2/epoch-0.pt').is_file()
    assert Path('results_save/ckpt2/epoch-2.pt').is_file()

    for batch_ind in [1, 2, 3]:
        dirname = 'results_save/train/epoch-2/batch-%d' % batch_ind
        for sample_ind in [1, 2, 3, 4]:
            ind = (batch_ind - 1) * 4 + sample_ind - 1
            for suffix in ['truth-cpu_y_%d', 'input-cpu_x_%d', 'output-cpu']:
                if '%' in suffix:
                    suffix = suffix % ind
                filename = 'sample-%d_%s.nii.gz' % (sample_ind, suffix)
                filename = Path(dirname, filename)
                assert filename.is_file()

    assert trainer.loss == -20300

    image = nib.load(Path('results_save/train/epoch-2/batch-1',
                          'sample-1_input-cpu_x_0.nii.gz')).get_fdata()
    assert image.tolist() == [0]
    image = nib.load(Path('results_save/train/epoch-2/batch-1',
                          'sample-3_truth-cpu_y_2.nii.gz')).get_fdata()
    assert image.tolist() == [3]

    image = nib.load(Path('results_save/train/epoch-2/batch-1',
                          'sample-2_input-cpu_x_1.nii.gz')).get_fdata()
    assert image.tolist() == [1]
    image = nib.load(Path('results_save/train/epoch-2/batch-2',
                          'sample-2_truth-cpu_y_5.nii.gz')).get_fdata()
    assert image.tolist() == [6]

    image = nib.load(Path('results_save/train/epoch-2/batch-3',
                          'sample-4_output-cpu.nii.gz')).get_fdata()
    assert image.tolist() == [-2926, -2926]

    image = nib.load(Path('results_save/valid/epoch-2/batch-4',
                          'sample-1_output-cpu.nii.gz')).get_fdata()
    assert image.tolist() == [-3582, -3582]
    print('successful')


if __name__ == '__main__':
    test_train_and_save()
