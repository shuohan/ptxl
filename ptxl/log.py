"""Classes to print and log training and validation progress.

"""
import numpy as np
from pathlib import Path
import warnings
from collections.abc import Iterable
from tqdm import trange, tqdm

from .abstract import Observer


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
        self.attrs = []
        self._writer = None

    def start(self):
        """Initializes the writer to log data."""
        names = self._counter.name
        fields = self._append_data(name, self.attrs)
        self._writer = Writer(self.filename, fields)
        self._writer.open()

    def close(self):
        """Closes the writer."""
        self._writer.close()

    def _update(self):
        contents = self._counter.index
        values = self.contents.get_values(self.attrs)
        contents = self._append_data(contents, values)
        self._writer.write_line(contents)

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


class Printer(Observer):
    """Abstract class to print the training or validation progress to stdout.

    Attributes:
        decimals (int): The number of decimals to print.

    """
    def __init__(self, decimals=4):
        super().__init__()
        self.decimals = decimals
        self.attrs = []

    def _update(self):
        num = self._counter.num
        index = self._counter.named_index
        num = [num] if not isinstance(num, Iterable) else index
        index = [index] if not isinstance(index, Iterable) else index
        index = ['/'.join([i.replace('-', ' '), n]) for i, n in zip(index, num)]
        values = self.contents.get_values(self.attrs)
        line = self._append_data(index, self.attrs, values)
        print(', '.join(line), flush=True)

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


class TqdmPrinter(Printer):
    """Uses tqdm to print training or validation progress.

    """
    def start(self):
        num = self._counter.num
        num = np.prod(num) if isinstance(num, Iterable) else num
        self._pbar = trange(num, dynamic_ncols=True, position=0)
        self._vbar = tqdm(bar_format='{desc}', dynamic_ncols=True, position=1)

    def _update(self):
        """Updates the tqdm progress bar."""
        values = self.contents.get_values(self.attrs)
        desc = ', '.join(self._append_data([], self.attrs, values))
        self._pbar.n = self._counter.index
        self._vbar.set_description(desc)
        self._pbar.refresh()
        self._vbar.refresh()

    def close(self):
        """Closes the tqdm progress bar."""
        self._vbar.close()
        self._pbar.close()


class MultiTqdmPrinter(TqdmPrinter):
    """Uses tqdm to print progress with multiple levels.

    """
    def start(self):
        super().update_on_train_start()
        assert isinstance(self._counter.name, Iterable)
        self._pbars = [trange(n, dynamic_ncols=True, position=i)
                       for i, n in enumerate(self._counter.num)]
        self._vbar = tqdm(bar_format='{desc}', dynamic_ncols=True,
                          position=len(self._counter.num))

    def _update(self):
        attrs = self.subject.abbrs
        values = self.contents.get_values(self.attrs)
        desc = ', '.join(self._append_data([], self.attrs, values))
        self._vbar.set_description(desc)
        self._vbar.refresh()
        for pbar, index in zip(self._pbars, self._counter.index):
            pbar.n = index
            pbar.refresh()

    def close(self):
        for pbar in self._pbars:
            pbar.close()
        self._vbar.close()
