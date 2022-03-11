"""Classes to print and log training and validation progress.

"""
import numpy as np
from pathlib import Path
import warnings
from collections.abc import Iterable
import shutil
from tqdm import tqdm

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
    def __init__(self, filename, attrs=[]):
        super().__init__()
        self.filename = filename
        self.attrs = attrs
        self._writer = None

    def start(self):
        """Initializes the writer to log data."""
        fields = self._append_data(self._get_counter_name(), self.attrs)
        self._writer = Writer(self.filename, fields)
        self._writer.open()

    def close(self):
        """Closes the writer."""
        self._writer.close()

    def _update(self):
        index = self._get_counter_index()
        values = self.contents.get_values(self.attrs)
        line = self._append_data(index, values)
        self._writer.write_line(line)

    def _get_counter_name(self):
        return self.contents.counter.name

    def _get_counter_index(self):
        return self.contents.counter.index1

    def _append_data(self, data_list, data_elem):
        """Appends an element to a list.

        Args:
            data_list (list): The list to append.
            data_elem (iterable or number): The element to append.

        Returns:
            list: The appended list.

        """
        if not isinstance(data_list, list):
            data_list = [data_list]
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
    def __init__(self, decimals=4, attrs=[]):
        super().__init__()
        self.decimals = decimals
        self.attrs = attrs

    def _update(self):
        num = self._get_counter_num()
        index = self._get_counter_index()
        num = [num] if not isinstance(num, Iterable) else num
        index = [index] if isinstance(index, str) else index
        index = ['/'.join([ind.replace('-', ' '), str(n)])
                 for ind, n in zip(index, num)]
        values = self.contents.get_values(self.attrs)
        line = self._append_data(index, self.attrs, values)
        print(', '.join(line), flush=True)

    def _get_counter_index(self):
        return self.contents.counter.named_index1

    def _get_counter_num(self):
        return self.contents.counter.num

    def _get_counter_name(self):
        return self.contents.counter.name

    def _create_epoch_pattern(self):
        """Creates the pattern to print epoch info."""
        pattern = '%%0%dd' % len(str(self.subject.num_epochs))
        num_epochs = pattern % self.subject.num_epochs
        self._epoch_pattern = 'epoch %s/%s' % (pattern, num_epochs)

    def _append_data(self, data_list, data_name, data_elem):
        """Appends a data element with its name into the list."""
        if isinstance(data_elem, Iterable):
            data_elem = [self._convert_num(d) for d in data_elem]
            data_elem = ['%s: %s' % (n, d) for n, d in zip(data_name, data_elem)]
            data_list.extend(data_elem)
        else:
            data_elem = '%s: %s' % (data_name, self._convert_num(data_elem))
            data_list.append(data_elem)
        return data_list

    def _convert_num(self, num):
        """Converts a number to scientific format."""
        if float(num).is_integer():
            return str(num)
        else:
            return ('%%.%de' % self.decimals) % num


class TqdmPrinterNoDesc(Printer):
    def __init__(self, loc_offset=0):
        super().__init__()
        self.loc_offset = loc_offset

    def start(self):
        num = self._get_counter_num()
        desc = self._get_counter_name()
        self._pbar = tqdm(total=num, dynamic_ncols=True, desc=desc,
                          position=self.loc_offset)
        self._num_cols = shutil.get_terminal_size(fallback=(120, 50)).columns - 1

    def _get_counter_num(self):
        return np.prod(self.contents.counter.num)

    def _get_counter_index(self):
        index = self.contents.counter.index1
        nums = self.contents.counter.num
        if isinstance(index, Iterable):
            index = np.ravel_multi_index(tuple(index), tuple(nums))
        return index

    def _update(self):
        counter_num = self._get_counter_num()
        counter_index = self._get_counter_index()
        step = counter_index - self._pbar.n
        if step > 0:
            self._pbar.update(step)
        elif step < 0:
            self._pbar.reset(counter_num)

    def close(self):
        self._pbar.close()


class TqdmPrinter(TqdmPrinterNoDesc):
    """Uses tqdm to print training or validation progress.

    """
    def __init__(self, decimals=4, attrs=[], loc_offset=0, num_desc_lines=1):
        super().__init__(loc_offset)
        self.decimals = decimals
        self.num_desc_lines = num_desc_lines
        self.attrs = self._divide_attrs(attrs)

    def _divide_attrs(self, attrs):
        q, r = divmod(len(attrs), self.num_desc_lines)
        nums = [q + 1] * r + [q] * (self.num_desc_lines - r)
        start_ind = 0
        results = list()
        for n in nums:
            stop_ind = start_ind + n
            part = attrs[start_ind : stop_ind]
            results.append(part)
            start_ind = stop_ind
        return results

    def start(self):
        num = self._get_counter_num()
        self._vbars = [tqdm(bar_format='{desc}', dynamic_ncols=True,
                            position=i + self.loc_offset)
                       for i in range(self.num_desc_lines)]
        self._pbar = tqdm(total=num, dynamic_ncols=True,
                          position=self.num_desc_lines + self.loc_offset)
        self._num_cols = shutil.get_terminal_size(fallback=(120, 50)).columns - 1

    def _update(self):
        """Updates the tqdm progress bar."""
        for vbar, attrs in zip(self._vbars, self.attrs):
            values = self.contents.get_values(attrs)
            desc = ', '.join(self._append_data([], attrs, values))
            desc = desc[:self._num_cols]
            vbar.set_description_str(desc)
            vbar.refresh()
        super()._update()

    def close(self):
        """Closes the tqdm progress bar."""
        for vbar in self._vbars:
            vbar.close()
        self._pbar.close()


class MultiTqdmPrinter(TqdmPrinter):
    """Uses tqdm to print progress with multiple levels.

    """
    def start(self):
        assert isinstance(self.contents.counter.name, Iterable)
        self._vbars = [tqdm(bar_format='{desc}', dynamic_ncols=True,
                            position=i + self.loc_offset)
                       for i in range(self.num_desc_lines)]

        num = self._get_counter_num()
        desc = self._get_counter_name()
        self._pbars = [tqdm(total=n, desc=d, dynamic_ncols=True,
                            position=i + self.num_desc_lines + self.loc_offset)
                       for i, (n, d) in enumerate(zip(num, desc))]
        self._num_cols = shutil.get_terminal_size(fallback=(120, 50)).columns - 1

    def _get_counter_num(self):
        return self.contents.counter.num

    def _get_counter_index(self):
        return self.contents.counter.index1

    def _get_counter_name(self):
        return self.contents.counter.name

    def _update(self):
        for vbar, attrs in zip(self._vbars, self.attrs):
            values = self.contents.get_values(attrs)
            desc = ', '.join(self._append_data([], attrs, values))
            desc = desc[:self._num_cols]
            vbar.set_description_str(desc)
            vbar.refresh()

        counter_num = self._get_counter_num()
        counter_index = self._get_counter_index()

        update_next = True
        for i in reversed(range(len(self._pbars))):
            if update_next:
                step = counter_index[i] - self._pbars[i].n
                if step > 0:
                    self._pbars[i].update(step)
                elif step < 0:
                    self._pbars[i].reset(counter_num[i])

                update_next = False
                if self._pbars[i].n == self._pbars[i].total:
                    update_next = True

    def close(self):
        for vbar in self._vbars:
            vbar.close()
        for pbar in self._pbars:
            pbar.close()
