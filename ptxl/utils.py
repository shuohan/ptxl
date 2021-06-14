"""Utility functions and classes.

"""
import itertools
from collections import namedtuple


NamedData = namedtuple('NamedData', ['name', 'data'])
"""Data with its name.

Args:
    name (str): The name of the data.
    data (numpy.ndarray): The data.
    
"""
class Counter_:
    """Abstract class to count indices.

    """
    @property
    def name(self):
        raise NotImplementedError

    @property
    def num(self):
        raise NotImplementedError

    @property
    def index0(self):
        raise NotImplementedError

    @property
    def named_index0(self):
        raise NotImplementedError

    @property
    def index1(self):
        raise NotImplementedError

    @property
    def named_index1(self):
        raise NotImplementedError

    @property
    def __iter__(self):
        raise NotImplementedError

    def has_reached_end(self):
        raise NotImplementedError


class Counter(Counter_):
    """Counts an index and resets it when the maximum number is reached.

    Attributes:
        name (str): The number of the counter.
        num (int): The maximum number of the index.

    """
    def __init__(self, name, num):
        self._name = name
        self._num = num
        self._current_index = -1
        self._next_index = 0
        self._template = self._get_template()

    def __iter__(self):
        self._current_index = -1
        self._next_index = 0
        return self

    def __next__(self):
        self._current_index = self._next_index
        if self._current_index < self.num:
            self._next_index += 1
            return self._current_index
        else:
            raise StopIteration

    @property
    def name(self):
        return self._name

    @property
    def num(self):
        return self._num

    def has_reached_end(self):
        return self.index1 == self.num

    @property
    def index0(self):
        return self._current_index

    @index0.setter
    def index0(self, index):
        self._current_index = index
        self._next_index = index + 1

    @property
    def named_index0(self):
        return self._template % self.index0

    @property
    def index1(self):
        return self._current_index + 1

    @property
    def named_index1(self):
        return self._template % self.index1

    def _get_template(self):
        return '-'.join([self.name, '%%0%dd' % len(str(self.num))])


class Counters(Counter_):
    def __init__(self, counters):
        self.counters = counters
        self._names = {c.name: i for i, c in enumerate(self.counters)}

    def __len__(self):
        return len(self.counters)

    @property
    def num(self):
        return [c.num for c in self.counters]

    @property
    def name(self):
        return [c.name for c in self.counters]

    @property
    def index1(self):
        return [c.index1 for c in self.counters]

    @property
    def named_index1(self):
        return [c.named_index1 for c in self.counters]

    def __iter__(self):
        return itertools.product(self.counters)

    def has_reached_end(self):
        return [c.has_reached_end() for c in self.counters]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.counters[self._names[key]]
        else:
            return self.counters[key]


class DataQueue:
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


def count_trainable_params(net):
    """Counts the trainable parameters of a network.

    Args:
        net (torch.nn.Module): The network to count.

    Returns:
        int: The number of trainable parameters.

    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
