"""Observer design pattern.

"""
from .utils import NamedData


class Observer:
    """Gets notified by :class:`Subject` to update its status.

    Note:
        This a minxin class. If a class inherts from multiple parent classes,
        this class should be put in front. If all parent class are mixins,
        the order does not matter.

        Any class inheriting from this class should also be a mixin in order to
        use multiple inheritance, i.e., it should implement

        >>> super().__init__(*args, **kwargs)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subject = None

    @property
    def subject(self):
        """The subject that is been observed."""
        return self._subject

    @subject.setter
    def subject(self, subject):
        self._check_subject_type(subject)
        self._subject = subject

    def _check_subject_type(self, subject):
        """Enforces the type of acceptable subjects here."""
        assert isinstance(subject, Subject)

    def update_on_train_start(self):
        """Update just before the training starts"""
        pass

    def update_on_epoch_start(self):
        """Update just before the current epoch starts"""
        pass

    def update_on_batch_start(self):
        """Update just before the current batch starts"""
        pass

    def update_on_batch_end(self):
        """Update right after the current batch ends"""
        pass

    def update_on_epoch_end(self):
        """Update right after the current epoch ends"""
        pass

    def update_on_train_end(self):
        """Update right after the training ends"""
        pass


class Subject:
    """An abstract class to notify registered :class:`Observer` for updates.

    Note:
        This a minxin class. If a class inherts from multiple parent classes,
        this class should be put in front. If all parent class are mixins,
        the order does not matter.

        Any class inheriting from this class should also be a mixin in order to
        use multiple inheritance, i.e., it should implement

        >>> super().__init__(*args, **kwargs)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers = list()

        self._values = dict()
        self._cpu_tensors = dict()
        self._cuda_tensors = dict()

    def get_value(self, value_attr):
        """Returns a value (e.g., loss).

        Args:
            value_attr (str): The attribute name of the value to return.

        Returns:
            float: The value.

        """
        return self._values[value_attr].item()

    def get_value_attrs(self):
        """Returns the attribute names of all available values."""
        return list(self._values.keys())

    def get_cpu_tensor_attrs(self):
        """Returns the attribute names of all available tensors on CPU."""
        return list(self._cpu_tensors.keys())

    def get_cuda_tensor_attrs(self):
        """Returns the attribute names of all available tensors on CUDA."""
        return list(self._cuda_tensors.keys())

    def get_tensor(self, tensor_attr, device='cpu'):
        """Returns a tensor on CPU or CUDA.

        Args:
            tensor_attr (str): The attribute name of the tensor to return.
            device (str): The the device of the tensor to return. It can only be
                "cpu" or "cuda".

        Returns:
            torch.Tensor/NamedData: The tensor.

        """
        if device == 'cpu':
            return self._get_cpu_tensor(tensor_attr)
        elif device == 'cuda':
            return self._get_cuda_tensor(tensor_attr)
        else:
            raise RuntimeError('device can only be "cpu" or "cuda".')

    def _get_cpu_tensor(self, tensor_attr):
        if tensor_attr in self._cpu_tensors:
            return self._cpu_tensors[tensor_attr]
        else:
            tensor = self._cuda_tensors[tensor_attr]
            if isinstance(tensor, NamedData):
                tensor = NamedData(tensor.name, tensor.data.detach().cpu())
            return tensor

    def _get_cuda_tensor(self, tensor_attr):
        if tensor_attr in self._cuda_tensors:
            return self._cuda_tensors[tensor_attr]
        else:
            tensor = self._cpu_tensors[tensor_attr]
            if isinstance(tensor, NamedData):
                tensor = NamedData(tensor.name, tensor.data.cuda())
            return tensor

    def _set_tensor_cpu(self, attr, tensor, name=None):
        """Add tensor with attr and name into the cpu collection."""
        if name is not None:
            tensor = NamedData(name=name, data=tensor)
        self._cpu_tensors[attr] = tensor

    def _set_tensor_cuda(self, attr, tensor, name=None):
        """Add tensor with attr and name into the cuda collection."""
        if name is not None:
            tensor = NamedData(name=name, data=tensor)
        self._cuda_tensors[attr] = tensor

    def register(self, observer):
        """Registers an observer to get notified.

        Args:
            observer (Observer): The observer to register.

        """
        observer.subject = self
        self._observers.append(observer)

    def remove(self, observer):
        """Removes an observer.

        Args:
            observer (Observer): The observer to remove. It has to be registered
                before.

        """
        self._observers.remove(observer)

    def notify_observers_on_train_start(self):
        """Notifies registered observers on the start of the training."""
        for observer in self._observers:
            observer.update_on_train_start()

    def notify_observers_on_epoch_start(self):
        """Notifies the observers on the start of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_start()

    def notify_observers_on_batch_start(self):
        """Notifies the observers on the start of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_start()

    def notify_observers_on_batch_end(self):
        """Notifies the observers on the end of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_end()

    def notify_observers_on_epoch_end(self):
        """Notifies the observers on the end of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_end()

    def notify_observers_on_train_end(self):
        """Notifies the observers on the end of the training."""
        for observer in self._observers:
            observer.update_on_train_end()

    @property
    def num_epochs(self):
        """Returns the number of epochs."""
        raise NotImplementedError

    @property
    def num_batches(self):
        """Returns the number batches per epoch."""
        raise NotImplementedError

    @property
    def batch_size(self):
        """Returns the number of samples per mini-batch."""
        raise NotImplementedError

    @property
    def epoch_ind(self):
        """Returns the current epoch index (1-based)."""
        raise NotImplementedError

    @property
    def batch_ind(self):
        """Returns the current batch index (1-based)."""
        raise NotImplementedError


class SubjectObserver(Subject, Observer):
    """A subject that is an observer at the same time.

    """
    def update_on_train_start(self):
        self.notify_observers_on_train_start()

    def update_on_epoch_start(self):
        self.notify_observers_on_epoch_start()

    def update_on_batch_start(self):
        self.notify_observers_on_batch_start()

    def update_on_batch_end(self):
        self.notify_observers_on_batch_end()

    def update_on_epoch_end(self):
        self.notify_observers_on_epoch_end()

    def update_on_train_end(self):
        self.notify_observers_on_train_end()
