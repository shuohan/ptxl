"""Classes to print and log training and validation progress.

"""
import numpy as np
from pathlib import Path
import warnings
from collections.abc import Iterable

from .observer import Observer, SubjectObserver


class DataQueue_:
    """This class implements a list that empties itself when full.

    Note:
        This class also supports adding arrays with the same shape.

    Args:
        maxlen (int): The maximum length of the list.

    """
    def __init__(self, maxlen):
        self._maxlen = maxlen
        self._buffer = [None] * self.maxlen
        self._ind = -1

    @property
    def maxlen(self):
        """Returns the maximum length of the list."""
        return self._maxlen

    def __len__(self):
        return self._ind + 1

    def put(self, value):
        """Adds a new element. Empties the list if full.

        Args:
            value (numpy.ndarray or number): The value to add. It will be
                converted to :class:`numpy.ndarray` when adding.

        Raises:
            ValueError: The value to add has different shape.

        """
        value = np.array(value)
        self._ind = 0 if self._ind == self.maxlen - 1 else self._ind + 1
        if self._ind > 0 and not self._shape_is_valid(value):
            raise ValueError('The value to add has different shape.')
        self._buffer[self._ind] = value

    def _shape_is_valid(self, value):
        """Checks if the newly added value has the same shape."""
        return value.shape == self._buffer[self._ind - 1].shape

    @property
    def current(self):
        """Returns current value as :class:`numpy.ndarray`."""
        if self._ind == -1:
            message = 'Buffer is empty. Return "nan" as the current value.'
            warnings.warn(message, RuntimeWarning)
            return np.nan
        else:
            return self._buffer[self._ind]

    @property
    def mean(self):
        """Returns the average aross all values as :class:`numpy.ndarray`."""
        if self._ind == -1:
            message = 'Buffer is empty. Return "nan" as the mean.'
            warnings.warn(message, RuntimeWarning)
            return np.nan
        else:
            return np.mean(self._buffer[:self._ind+1], axis=0)

    @property
    def all(self):
        """Returns all values.

        Note:
            The 0th axis is the stacking axis.

        Returns:
            numpy.ndarray: The stacked values.

        """
        if self._ind == -1:
            message = 'Buffer is empty. Return numpy.array([]) as all values.'
            warnings.warn(message, RuntimeWarning)
            return np.array(list())
        else:
            return np.stack(self._buffer[:self._ind+1], axis=0)


class DataQueue(SubjectObserver):
    """Wrapper of :class:`DataQueue_` to add observer and subject functions.

    Attributes:
        attrs (str or list[str]): The name(s) of the attribute(s) of the
            subject to monitor. When it is iterable, its length should match the
            shape of the data to add.
        abbrs (str or list[str]): The abbreviation(s) of the attribute(s).

    """
    def __init__(self, attrs, abbrs=None):
        super().__init__()
        self.attrs = attrs
        self.abbrs = self.attrs if abbrs is None else abbrs
        assert len(self.abbrs) == len(self.attrs)
        self._queue = None

    def update_on_train_start(self):
        self._queue = DataQueue_(self.num_batches)
        super().update_on_train_start()

    @property
    def batch_size(self):
        return self.subject.batch_size

    def __len__(self):
        return len(self._queue)

    def put(self, value):
        self._queue.put(value)

    @property
    def mean(self):
        return self._queue.mean

    @property
    def current(self):
        return self._queue.current

    @property
    def all(self):
        return self._queue.all

    @property
    def num_epochs(self):
        return self.subject.num_epochs

    @property
    def num_batches(self):
        return self.subject.num_batches

    @property
    def epoch_ind(self):
        return self.subject.epoch_ind

    @property
    def batch_ind(self):
        return self.subject.batch_ind

    def update_on_batch_end(self):
        if isinstance(self.attrs, list):
            value = [getattr(self.subject, n) for n in self.attrs]
        else:
            value = getattr(self.subject, self.attrs)
        self.put(value)
        super().update_on_batch_end()


class Writer:
    """Write contents into a .csv file.

    Attributes:
        filename (pathlib.Path): The filename of the output file.
        fields (iterable[str]): The names of the fields.

    Args:
        filename (str or pathlib.Path): The filename of the output file.

    """
    def __init__(self, filename, fields=[]):
        self.filename = Path(filename)
        self.fields = fields
        self._file = None

    def open(self):
        """Opens the file and make directory."""
        if self.filename.is_file():
            message = 'The file %s exists. Append new contents.' % self.filename
            warnings.warn(message, RuntimeWarning)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filename, 'a')
        self._write_header()

    def is_open(self):
        """Checks if the file is opened."""
        return self._file is not None and not self._file.closed

    def _write_header(self):
        header = ','.join(self.fields) + '\n'
        self._file.write(header)

    def write_line(self, data):
        """Writes a line into the file.

        Args:
            data (iterable): The contents to write. The order should be the same
                with :attr:`fields`.

        """
        line = ','.join(['%g' % d for d in data]) + '\n'
        self._file.write(line)
        self._file.flush()

    def close(self):
        self._file.close()


class Logger(Observer):
    """Abstract to log training or validation progress.

    Attributes:
        subject (DataQueue): The subject to extract data from.

    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._writer = None

    def _check_subject_type(self, subject):
        assert isinstance(subject, DataQueue)

    def update_on_train_end(self):
        """Closes the writer."""
        self._writer.close()

    def _append_data(self, data_list, data_elem):
        """Appends an element to a list.

        Args:
            data_list (list): The list to append.
            data_elem (iterable or number): The element to append.

        Returns:
            list: The appended list.

        """
        if isinstance(data_elem, list):
            data_list.extend(data_elem)
        else:
            data_list.append(data_elem)
        return data_list


class BatchLogger(Logger):
    """Logs training or validation progress at the end of each batch.

    """
    def update_on_train_start(self):
        """Initializes the writer to log data."""
        fields = self._append_data(['epoch', 'batch'], self.subject.abbrs)
        self._writer = Writer(self.filename, fields)
        self._writer.open()

    def update_on_batch_end(self):
        """Logs the data into the file."""
        contents = [self.subject.epoch_ind, self.subject.batch_ind]
        contents = self._append_data(contents, self.subject.current.tolist())
        self._writer.write_line(contents)


class EpochLogger(Logger):
    """Logs training or validation progress at the end of each epoch.

    """
    def update_on_train_start(self):
        """Initializes the writer to log data."""
        fields = self._append_data(['epoch'], self.subject.abbrs)
        self._writer = Writer(self.filename, fields)
        self._writer.open()

    def update_on_epoch_end(self):
        """Logs the data into the file."""
        contents = [self.subject.epoch_ind]
        contents = self._append_data(contents, self.subject.mean.tolist())
        self._writer.write_line(contents)


class Printer(Observer):
    """Abstract class to print the training or validation progress to stdout.

    Attributes:
        decimals (int): The number of decimals to print.
        subject (DataQueue): The subject to print data from.

    """
    def __init__(self, decimals=4):
        super().__init__()
        self.decimals = decimals

    def update_on_train_start(self):
        """Initializes printing message."""
        self._create_epoch_pattern()

    def _create_epoch_pattern(self):
        """Creates the pattern to print epoch info."""
        pattern = '%%0%dd' % len(str(self.subject.num_epochs))
        num_epochs = pattern % self.subject.num_epochs
        self._epoch_pattern = 'epoch %s/%s' % (pattern, num_epochs)

    def _append_data(self, data_list, data_name, data_elem):
        """Appends a data element with its name into the list."""
        if isinstance(data_elem, Iterable):
            data_elem = [self._convert_num(d) for d in data_elem]
            data_elem = ['%s %s' % (n, d) for n, d in zip(data_name, data_elem)]
            data_list.extend(data_elem)
        else:
            data_elem = '%s %s' % (data_name, self._convert_num(data_elem))
            data_list.append(data_elem)
        return data_list

    def _convert_num(self, num):
        """Converts a number to scientific format."""
        return ('%%.%de' % self.decimals) % num


class EpochPrinter(Printer):
    """Prints training or validation progress at the end of each epoch.

    Attributes:
        print_sep (bool): Print "------" after each message if True.

    """
    def __init__(self, decimals=4, print_sep=True):
        super().__init__(decimals)
        self.print_sep = print_sep

    def update_on_epoch_end(self):
        """Prints the progress message at the end of each epoch."""
        attrs = self.subject.abbrs
        data = self.subject.mean.tolist()
        line = [self._epoch_pattern % self.subject.epoch_ind]
        line = self._append_data(line, attrs, data)
        print(', '.join(line), flush=True)
        if self.print_sep:
            print('------')


class BatchEpochPrinter(EpochPrinter):
    """Prints progress at the end of each mini-batch and each of epoch.

    """
    def update_on_train_start(self):
        self._create_batch_pattern()
        super().update_on_train_start()

    def _create_batch_pattern(self):
        """Creates the pattern to print batch info."""
        pattern = '%%0%dd' % len(str(self.subject.num_batches))
        num_batches = pattern % self.subject.num_batches
        self._batch_pattern = 'batch %s/%s' % (pattern, num_batches)

    def update_on_batch_end(self):
        """Prints the progress message at the end of each batch."""
        attrs = self.subject.abbrs
        data = self.subject.current.tolist()
        line = [self._epoch_pattern % self.subject.epoch_ind,
                self._batch_pattern % self.subject.batch_ind]
        line = self._append_data(line, attrs, data)
        print(', '.join(line), flush=True)
