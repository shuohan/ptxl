"""Observer design pattern.

"""
from .utils import NamedData


class Contents:
    """Training contents.

    This class implements the Subject in the observer design pattern. It holds
    the network model, optimizer, tensors (input, output, truth, etc.), and some
    values (loss values) during training.

    """
    def __init__(self, model, optim, counter):
        self.model = model
        self.optim = optim
        self.counter = counter

        self._values = dict()
        self._tensors_cpu = dict()
        self._tensors_cuda = dict()
        self._observers = list()

    def get_model_state_dict(self):
        """Returns the state dict of the network(s)."""
        return self.model.state_dict()

    def get_optim_state_dict(self):
        """Returns the state dict of the optimizer(s)."""
        return self.optim.state_dict()

    def load_state_dicts(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def set_value(self, value_attr, value):
        """Sets a value.

        Args:
            value_attr (str): The attribute name of this value.
            value (float): The value to set.

        """
        self._values[value_attr] = value

    def get_value(self, value_attr):
        """Returns a value (e.g., loss).

        Args:
            value_attr (str): The attribute name of the value.

        Returns:
            float: The value.

        """
        return self._values[value_attr]

    def get_values(self, value_attrs):
        """Returns values (e.g., losses).

        Args:
            value_attrs (iterable[str]): The attribute names of the values.

        Returns:
            list[float]: The values.

        """
        return [self._values[a] for a in value_attrs]

    def get_value_attrs(self):
        """Returns the attribute names of all available values."""
        return list(self._values.keys())

    def get_tensor_attrs(self):
        return self.get_tensor_attrs_cpu() + self.get_tensor_attrs_cuda()

    def get_tensor_attrs_cpu(self):
        """Returns the attribute names of all available tensors on CPU."""
        return list(self._tensors_cpu.keys())

    def get_tensor_attrs_cuda(self):
        """Returns the attribute names of all available tensors on CUDA."""
        return list(self._tensors_cuda.keys())

    def get_tensor(self, tensor_attr, device='cuda'):
        """Returns a tensor on CPU or CUDA.

        Args:
            tensor_attr (str): The attribute name of the tensor to return.
            device (str): The the device of the tensor to return. It can only be
                "cpu" or "cuda".

        Returns:
            torch.Tensor/NamedData: The tensor.

        """
        if device == 'cpu':
            return self.get_tensor_cpu(tensor_attr)
        elif device == 'cuda':
            return self.get_tensor_cuda(tensor_attr)
        else:
            raise RuntimeError('device can only be "cpu" or "cuda".')

    def get_tensor_cpu(self, tensor_attr):
        if tensor_attr in self._tensors_cpu:
            return self._tensors_cpu[tensor_attr]
        else:
            tensor = self._tensors_cuda[tensor_attr]
            if isinstance(tensor, NamedData):
                tensor = NamedData(tensor.name, tensor.data.detach().cpu())
            else:
                tensor = tensor.detach().cpu()
            return tensor

    def get_tensor_cuda(self, tensor_attr):
        if tensor_attr in self._tensors_cuda:
            return self._tensors_cuda[tensor_attr]
        else:
            tensor = self._tensors_cpu[tensor_attr]
            if isinstance(tensor, NamedData):
                tensor = NamedData(tensor.name, tensor.data.cuda())
            else:
                tensor = tensor.cuda()
            return tensor

    def set_tensor(self, attr, tensor, name=None, device='cuda'):
        if device == 'cuda':
            self.set_tensor_cuda(attr, tensor, name)
        elif device == 'cpu':
            self.set_tensor_cpu(attr, tensor, name)
        else:
            raise RuntimeError('device can only be "cpu" or "cuda".')

    def set_tensor_cpu(self, attr, tensor, name=None):
        """Add tensor with attr and name into the cpu collection."""
        if name:
            tensor = NamedData(name=name, data=tensor)
        self._tensors_cpu[attr] = tensor

    def set_tensor_cuda(self, attr, tensor, name=None):
        """Add tensor with attr and name into the cuda collection."""
        if name:
            tensor = NamedData(name=name, data=tensor)
        self._tensors_cuda[attr] = tensor

    def register(self, observer):
        """Registers an observer to get notified.

        Args:
            observer (Observer): The observer to register.

        """
        observer.set_contents(self)
        self._observers.append(observer)

    def remove(self, observer):
        """Removes an observer.

        Args:
            observer (Observer): The observer to remove. It has to be registered
                before.

        """
        self._observers.remove(observer)

    def start_observers(self):
        for ob in self._observers:
            ob.start()

    def notify_observers(self):
        for ob in self._observers:
            ob.update()

    def close_observers(self):
        for ob in self._observers:
            ob.close()


class Observer:
    """Gets notified by :class:`Contents` to update its status.

    Args:
        step (int): The update step size.

    """
    def __init__(self):
        self._contents = None

    @property
    def contents(self):
        """The contents that is been observed."""
        return self._contents

    def set_contents(self, contents):
        self._check_contents_type(contents)
        self._contents = contents

    def _check_contents_type(self, contents):
        """Enforces the type of acceptable contents here."""
        assert isinstance(contents, Contents)

    def start(self):
        """Does some initialization after :meth:`contents` is registered."""
        pass

    def close(self):
        """Cleans up itself after all operations are finished."""
        pass

    def update(self):
        """Updates itself."""
        if self._needs_to_update():
            self._update()

    def _needs_to_update(self):
        return True

    def _update(self):
        pass
