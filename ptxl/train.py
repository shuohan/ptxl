"""Classes to train, validate, and evaluate an algorithm.

"""
import torch

from .observer import Subject, SubjectObserver, Observer
from .utils import NamedData


class Trainer(Subject):
    """Abstract class to train an algorithm.

    Args:
        num_epochs (int): The number of epochs.
    
    """
    def __init__(self, num_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_epochs = num_epochs
        self._epoch_ind = -1

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def epoch_ind(self):
        return self._epoch_ind + 1

    def set_epoch_ind(self, ind):
        self._epoch_ind = ind - 1

    def get_model_state_dict(self):
        """Returns the state_dict of the training model(s)."""
        raise NotImplementedError

    def get_optim_state_dict(self):
        """Returns the state_dict of the optimizer(s)."""
        raise NotImplementedError


class Validator(SubjectObserver):
    """Abstract class to validate an algorithm.

    Attributes:
        step (int): Validate the network every this number of epochs.
    
    """
    def __init__(self, step):
        super().__init__()
        self.step = step

    def _check_subject_type(self, subject):
        assert isinstance(subject, Trainer)

    @property
    def num_epochs(self):
        return self.subject.num_epochs

    @property
    def epoch_ind(self):
        return self.subject.epoch_ind

    @property
    def batch_size(self):
        return self.subject.batch_size

    def update_on_batch_start(self):
        pass

    def update_on_batch_end(self):
        pass


class Evaluator(Observer):
    """Abstract class to evaluate an algorithm.

    Attributes:
        eval_funcs (dict): The :class:`str` name and function pairs of the
            evaluation functions. The functions should be defined as
            ``func(output, truth)``.
    
    """
    def __init__(self, eval_funcs):
        super().__init__()
        self.eval_funcs = eval_funcs

    def _check_subject_type(self, subject):
        assert isinstance(subject, Trainer) or isinstance(subject, Validator)

    @property
    def batch_size(self):
        return self.subject.batch_size


class SimpleMixin:
    """Mixin class for :class:`SimpleTrainer` and :class:`SimpleValidator`.

    Attributes:
        dataloader (torch.utils.data.DataLoader): Yields training data.
    
    """
    def __init__(self, *args, dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader
        self._input_cpu = None
        self._input_cuda = None
        self._output_cpu = None
        self._output_cuda = None
        self._truth_cpu = None
        self._truth_cuda = None
        self._input_names = None
        self._truth_names = None
        self._batch_ind = -1
        self._value = None

    @property
    def num_batches(self):
        return len(self.dataloader)

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def batch_ind(self):
        return self._batch_ind + 1

    @property
    def loss(self):
        """Returns the loss value.

        Returns:
            number of iterable[number]: The loss values to keep track of.

        """
        return self._loss.item()

    @property
    def input_cpu(self):
        """Returns the network input as :class:`torch.Tensor` on CPU."""
        input_cpu = self._input_cpu 
        if self._input_names is not None:
            input_cpu = NamedData(self._input_names, input_cpu)
        return input_cpu

    @property
    def input_cuda(self):
        """Returns the network input as :class:`torch.Tensor` on CUDA."""
        return self._input_cuda

    @property
    def output_cpu(self):
        """Returns the network output as :class:`torch.Tensor` on CPU.

        Note:
            :attr:`output_cpu` is only converted from GPU when it is ``None``.
            This means that the buffer variable ``self._output_cpu`` should be
            cleared out in each iteration. The benefit of this design is that
            after back-propgation, this variable only need to be converted once
            for access from multiple threads.

        """
        if self._output_cpu is None:
            self._output_cpu = self._output_cuda.detach().cpu()
        return self._output_cpu

    @property
    def output_cuda(self):
        """Returns the network output as :class:`torch.Tensor` on CUDA."""
        return self._output_cuda

    @property
    def truth_cpu(self):
        """Returns the network truth as :class:`torch.Tensor` on CPU."""
        truth_cpu = self._truth_cpu 
        if self._truth_names is not None:
            truth_cpu = NamedData(self._truth_names, truth_cpu)
        return truth_cpu

    @property
    def truth_cuda(self):
        """Returns the network truth as :class:`torch.Tensor` on CUDA."""
        return self._truth_cuda

    def _empty(self):
        """Empties temporary tensors."""
        self._input_cpu = None
        self._input_cuda = None
        self._output_cpu = None
        self._output_cuda = None
        self._truth_cpu = None
        self._truth_cuda = None
        self._input_names = None
        self._truth_names = None

    def _parse_input(self, data):
        """Extracts input tensor names, on cpu, and on cuda."""
        self._input_names, self._input_cpu, self._input_cuda = self._parse(data)

    def _parse_truth(self, data):
        """Extracts truth tensor names, on cpu, and on cuda."""
        self._truth_names, self._truth_cpu, self._truth_cuda = self._parse(data)

    def _parse(self, data):
        """Extracts tensor names, on cpu, and on cuda."""
        name = None
        if isinstance(data, NamedData):
            name = data.name
            data = data.data
        data_cuda = data.cuda()
        return name, data, data_cuda


class SimpleTrainer(SimpleMixin, Trainer):
    """Trains an algorithm with one network, optim, and loss function.

    Attributes:
        net (torch.nn.Module): The network to train.
        optim (torch.optim.Optimizer): The network optimizer.
        loss_func (torch.nn.Module): Calculates the loss.

    """
    def __init__(self, net, optim, dataloader, loss_func, num_epochs=100):
        super().__init__(num_epochs, dataloader=dataloader)
        self.net = net
        self.optim = optim
        self.loss_func = loss_func

    def train(self):
        """Trains the network."""
        self.notify_observers_on_train_start()
        for self._epoch_ind in range(self.num_epochs):
            self.notify_observers_on_epoch_start()
            for self._batch_ind, data in enumerate(self.dataloader):
                self._empty() # NOTE: self.output_cpu is wrong without this
                self._parse_input(data[0])
                self._parse_truth(data[1])
                self.notify_observers_on_batch_start()
                self._train_on_batch()
                self.notify_observers_on_batch_end()
            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()

    def _train_on_batch(self):
        """Trains the network on each mini-batch.
        
        """
        self.optim.zero_grad()
        self._output_cuda = self.net(self._input_cuda)
        self._loss = self.loss_func(self._output_cuda, self._truth_cuda)
        self._loss.backward()
        self.optim.step()

    def get_model_state_dict(self):
        return self.net.state_dict()

    def get_optim_state_dict(self):
        return self.optim.state_dict()


class SimpleValidator(SimpleMixin, Validator):
    """Validates an algorithm with one network, optim, and loss_func.

    """
    def __init__(self, dataloader, step=10):
        super().__init__(step, dataloader=dataloader)

    def _check_subject_type(self, subject):
        assert isinstance(subject, SimpleTrainer)

    def update_on_epoch_end(self):
        if (self.epoch_ind % self.step) == 0:
            self.notify_observers_on_epoch_start()
            with torch.no_grad():
                self.subject.net.eval() 
                for self._batch_ind, data in enumerate(self.dataloader):
                    self._empty() # NOTE: self.output_cpu is wrong without this
                    self._parse_input(data[0])
                    self._parse_truth(data[1])
                    self.notify_observers_on_batch_start()
                    self._validate_on_batch()
                    self.notify_observers_on_batch_end()
            self.notify_observers_on_epoch_end()

    def _validate_on_batch(self):
        """Validates the network on a batch."""
        self._output_cuda = self.subject.net(self._input_cuda)
        self._loss = self.subject.loss_func(self._output_cuda, self._truth_cuda)


class SimpleEvaluator(Evaluator):
    """Evaluates an algorithm with one network, optim, and loss function.
    
    """
    def _check_subject_type(self, subject):
        assert isinstance(subject, SimpleValidator) \
            or isinstance(subject, SimpleTrainer)

    def update_on_batch_end(self):
        output_cuda = self.subject.output_cuda
        truth_cuda = self.subject.truth_cuda
        with torch.no_grad():
            for name, func in self.eval_funcs.items():
                setattr(self, name, func(output_cuda, truth_cuda))
