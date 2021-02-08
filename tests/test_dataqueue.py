#!/usr/bin/env python

import numpy as np

from ptxl.log import DataQueue_, DataQueue
from ptxl.observer import Subject


class _Subject(Subject):

    def __init__(self):
        super().__init__()
        self.data1 = 0
        self.data2 = 10

    def run(self):
        self.notify_observers_on_train_start()
        for self._epoch_ind in range(self.num_epochs):
            self.notify_observers_on_epoch_start()
            for self._batch_ind in range(self.num_batches):
                self.notify_observers_on_batch_start()
                self.data1 += 1
                self.data2 += 1
                self.notify_observers_on_batch_end()
            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()
        
    @property
    def batch_size(self): 
        return 3

    @property
    def num_epochs(self):
        return 2

    @property
    def num_batches(self):
        return 10

    @property
    def epoch_ind(self):
        return self._epoch_ind + 1

    @property
    def batch_ind(self):
        return self._batch_ind + 1


def test_queue_scalar():
    queue = DataQueue_(3)
    assert len(queue) == 0
    assert np.isnan(queue.current)
    assert np.isnan(queue.mean)
    assert queue.all.size == 0

    queue.put(0.1)
    assert len(queue) == 1
    assert queue.current == 0.1
    assert queue.mean == 0.1
    assert queue.all == [0.1]

    queue.put(0.2)
    assert len(queue) == 2
    assert queue.current == 0.2
    assert np.isclose(queue.mean, 0.15)
    assert queue.all.tolist() == [0.1, 0.2]

    queue.put(0.3)
    assert len(queue) == 3
    assert queue.current == 0.3
    assert np.isclose(queue.mean, 0.20)
    assert queue.all.tolist() == [0.1, 0.2, 0.3]

    queue.put(0.4)
    assert len(queue) == 1
    assert queue.current == 0.4
    assert queue.mean == 0.4
    assert queue.all.tolist() == [0.4]

    print('test queue scalar.')


def test_queue_array():
    queue = DataQueue_(3)
    assert len(queue) == 0
    assert np.isnan(queue.current)
    assert np.isnan(queue.mean)
    assert queue.all.size == 0

    queue.put(np.array([[1, 2, 3], [4, 5, 6]]))
    assert len(queue) == 1
    assert (queue.current == np.array([[1, 2, 3], [4, 5, 6]])).all()
    assert (queue.mean == np.array([[1, 2, 3], [4, 5, 6]])).all()
    all = np.vstack(queue.all)
    assert (all == np.vstack(np.array([[1, 2, 3], [4, 5, 6]]))).all()

    queue.put(np.array([[11, 12, 13], [14, 15, 16]]))
    assert len(queue) == 2
    assert (queue.current == np.array([[11, 12, 13], [14, 15, 16]])).all()
    assert np.isclose(queue.mean, np.array([[6, 7, 8], [9, 10, 11]])).all()
    all = np.stack(queue.all, axis=0)
    ref = np.stack([np.array([[1, 2, 3], [4, 5, 6]]),
                    np.array([[11, 12, 13], [14, 15, 16]])], axis=0)
    assert (all == ref).all()

    queue.put(np.array([[21, 22, 23], [24, 25, 26]]))
    assert len(queue) == 3
    assert (queue.current == np.array([[21, 22, 23], [24, 25, 26]])).all()
    ref = np.array([[11, 12, 13], [14, 15, 16]])
    assert np.isclose(queue.mean, ref).all()
    all = np.stack(queue.all, axis=0)
    ref = np.stack([np.array([[1, 2, 3], [4, 5, 6]]),
                    np.array([[11, 12, 13], [14, 15, 16]]),
                    np.array([[21, 22, 23], [24, 25, 26]])], axis=0)
    assert (all == ref).all()

    queue.put(np.array([[31, 32, 33], [34, 35, 36]]))
    assert len(queue) == 1
    assert (queue.current == np.array([[31, 32, 33], [34, 35, 36]])).all()
    assert (queue.mean == np.array([[31, 32, 33], [34, 35, 36]])).all()
    all = np.vstack(queue.all)
    assert (all == np.vstack(np.array([[31, 32, 33], [34, 35, 36]]))).all()
    print('test queue array.')


def test_wrong_shape():
    queue = DataQueue_(3)
    queue.put(1)
    try:
        queue.put(np.array([1, 2, 3]))
    except ValueError:
        print('Wrong shape')
        assert True
    print('test wrong shape successful.')


def test_dataqueue():
    subject = _Subject()
    queue1 = DataQueue('data1')
    queue2 = DataQueue(['data1', 'data2'])
    subject.register(queue1)
    subject.register(queue2)
    subject.run() 

    assert queue1.batch_size == 3
    assert queue1.mean == 15.5
    assert queue1.current == 20
    assert queue1.all.tolist() == list(range(11, 21))
    assert len(queue1) == 10
    assert queue1.num_epochs == 2

    assert queue2.batch_size == 3
    assert queue2.mean.tolist() == [15.5, 25.5]
    assert queue2.current.tolist() == [20, 30]
    assert len(queue2) == 10
    assert queue2.all.T.tolist() == [list(range(11, 21)), list(range(21, 31))]
    assert queue2.num_epochs == 2

    print('test data queue.')


if __name__ == '__main__':
    test_queue_scalar()
    test_queue_array()
    test_wrong_shape()
    test_dataqueue()
