#!/usr/bin/env python

import sys
from io import StringIO
from pathlib import Path
import shutil

from ptxl.log import DataQueue
from ptxl.log import BatchEpochPrinter, EpochPrinter
from ptxl.log import BatchLogger, EpochLogger

from test_dataqueue import _Subject


class Capturing(list):
    # https://stackoverflow.com/questions/16571150/
    # how-to-capture-stdout-output-from-a-python-function-call
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def test_log():
    if Path('results_log').is_dir():
        shutil.rmtree('results_log')
    subject = _Subject()
    queue1 = DataQueue('data1')
    printer11 = BatchEpochPrinter()
    printer12 = EpochPrinter()
    logger11 = BatchLogger('results_log/queue1.csv')
    logger12 = EpochLogger('results_log/queue2.csv')
    subject.register(queue1)
    queue1.register(printer11)
    queue1.register(printer12)
    queue1.register(logger11)
    queue1.register(logger12)
    with Capturing() as output:
        subject.run()
    ref = ['epoch 1/2, batch 01/10, data1 1.0000e+00',
           'epoch 1/2, batch 02/10, data1 2.0000e+00',
           'epoch 1/2, batch 03/10, data1 3.0000e+00',
           'epoch 1/2, batch 04/10, data1 4.0000e+00',
           'epoch 1/2, batch 05/10, data1 5.0000e+00',
           'epoch 1/2, batch 06/10, data1 6.0000e+00',
           'epoch 1/2, batch 07/10, data1 7.0000e+00',
           'epoch 1/2, batch 08/10, data1 8.0000e+00',
           'epoch 1/2, batch 09/10, data1 9.0000e+00',
           'epoch 1/2, batch 10/10, data1 1.0000e+01',
           'epoch 1/2, data1 5.5000e+00',
           '------',
           'epoch 1/2, data1 5.5000e+00',
           '------',
           'epoch 2/2, batch 01/10, data1 1.1000e+01',
           'epoch 2/2, batch 02/10, data1 1.2000e+01',
           'epoch 2/2, batch 03/10, data1 1.3000e+01',
           'epoch 2/2, batch 04/10, data1 1.4000e+01',
           'epoch 2/2, batch 05/10, data1 1.5000e+01',
           'epoch 2/2, batch 06/10, data1 1.6000e+01',
           'epoch 2/2, batch 07/10, data1 1.7000e+01',
           'epoch 2/2, batch 08/10, data1 1.8000e+01',
           'epoch 2/2, batch 09/10, data1 1.9000e+01',
           'epoch 2/2, batch 10/10, data1 2.0000e+01',
           'epoch 2/2, data1 1.5500e+01',
           '------',
           'epoch 2/2, data1 1.5500e+01',
           '------']
    assert ref == output
    with open('results_log/queue1.csv') as csv_file:
        log1 = [l.strip() for l in csv_file.readlines()]
    ref1 = ['epoch,batch,data1',
            '1,1,1',
            '1,2,2',
            '1,3,3',
            '1,4,4',
            '1,5,5',
            '1,6,6',
            '1,7,7',
            '1,8,8',
            '1,9,9',
            '1,10,10',
            '2,1,11',
            '2,2,12',
            '2,3,13',
            '2,4,14',
            '2,5,15',
            '2,6,16',
            '2,7,17',
            '2,8,18',
            '2,9,19',
            '2,10,20']
    assert log1 == ref1
    with open('results_log/queue2.csv') as csv_file:
        log2 = [l.strip() for l in csv_file.readlines()]
    ref2 = ['epoch,data1',
            '1,5.5',
            '2,15.5']
    assert log2 == ref2

    subject = _Subject()
    queue2 = DataQueue(['data1', 'data2'])
    printer21 = BatchEpochPrinter()
    printer22 = EpochPrinter()
    logger21 = BatchLogger('results_log/queue1.csv')
    logger22 = EpochLogger('results_log/queue2.csv')
    subject.register(queue2)
    queue2.register(printer21)
    queue2.register(printer22)
    queue2.register(logger21)
    queue2.register(logger22)
    with Capturing() as output:
        subject.run()
    ref = ['epoch 1/2, batch 01/10, data1 1.0000e+00, data2 1.1000e+01',
           'epoch 1/2, batch 02/10, data1 2.0000e+00, data2 1.2000e+01',
           'epoch 1/2, batch 03/10, data1 3.0000e+00, data2 1.3000e+01',
           'epoch 1/2, batch 04/10, data1 4.0000e+00, data2 1.4000e+01',
           'epoch 1/2, batch 05/10, data1 5.0000e+00, data2 1.5000e+01',
           'epoch 1/2, batch 06/10, data1 6.0000e+00, data2 1.6000e+01',
           'epoch 1/2, batch 07/10, data1 7.0000e+00, data2 1.7000e+01',
           'epoch 1/2, batch 08/10, data1 8.0000e+00, data2 1.8000e+01',
           'epoch 1/2, batch 09/10, data1 9.0000e+00, data2 1.9000e+01',
           'epoch 1/2, batch 10/10, data1 1.0000e+01, data2 2.0000e+01',
           'epoch 1/2, data1 5.5000e+00, data2 1.5500e+01',
           '------',
           'epoch 1/2, data1 5.5000e+00, data2 1.5500e+01',
           '------',
           'epoch 2/2, batch 01/10, data1 1.1000e+01, data2 2.1000e+01',
           'epoch 2/2, batch 02/10, data1 1.2000e+01, data2 2.2000e+01',
           'epoch 2/2, batch 03/10, data1 1.3000e+01, data2 2.3000e+01',
           'epoch 2/2, batch 04/10, data1 1.4000e+01, data2 2.4000e+01',
           'epoch 2/2, batch 05/10, data1 1.5000e+01, data2 2.5000e+01',
           'epoch 2/2, batch 06/10, data1 1.6000e+01, data2 2.6000e+01',
           'epoch 2/2, batch 07/10, data1 1.7000e+01, data2 2.7000e+01',
           'epoch 2/2, batch 08/10, data1 1.8000e+01, data2 2.8000e+01',
           'epoch 2/2, batch 09/10, data1 1.9000e+01, data2 2.9000e+01',
           'epoch 2/2, batch 10/10, data1 2.0000e+01, data2 3.0000e+01',
           'epoch 2/2, data1 1.5500e+01, data2 2.5500e+01',
           '------',
           'epoch 2/2, data1 1.5500e+01, data2 2.5500e+01',
           '------']
    assert ref == output

    with open('results_log/queue1.csv') as csv_file:
        log1 = [l.strip() for l in csv_file.readlines()]
    ref1 = ['epoch,batch,data1',
            '1,1,1',
            '1,2,2',
            '1,3,3',
            '1,4,4',
            '1,5,5',
            '1,6,6',
            '1,7,7',
            '1,8,8',
            '1,9,9',
            '1,10,10',
            '2,1,11',
            '2,2,12',
            '2,3,13',
            '2,4,14',
            '2,5,15',
            '2,6,16',
            '2,7,17',
            '2,8,18',
            '2,9,19',
            '2,10,20',
            'epoch,batch,data1,data2',
            '1,1,1,11',
            '1,2,2,12',
            '1,3,3,13',
            '1,4,4,14',
            '1,5,5,15',
            '1,6,6,16',
            '1,7,7,17',
            '1,8,8,18',
            '1,9,9,19',
            '1,10,10,20',
            '2,1,11,21',
            '2,2,12,22',
            '2,3,13,23',
            '2,4,14,24',
            '2,5,15,25',
            '2,6,16,26',
            '2,7,17,27',
            '2,8,18,28',
            '2,9,19,29',
            '2,10,20,30']
    assert log1 == ref1
    with open('results_log/queue2.csv') as csv_file:
        log2 = [l.strip() for l in csv_file.readlines()]
    ref2 = ['epoch,data1',
            '1,5.5',
            '2,15.5',
            'epoch,data1,data2',
            '1,5.5,15.5',
            '2,15.5,25.5']
    assert log2 == ref2

    print('successfull')


if __name__ == '__main__':
    test_log()
