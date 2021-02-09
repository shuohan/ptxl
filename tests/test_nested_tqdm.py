#!/usr/bin/env python

from tqdm.auto import trange
from time import sleep

for i in trange(4, desc='1st loop'):
    for k in trange(50, desc='3rd loop', leave=False):
        sleep(0.01)
