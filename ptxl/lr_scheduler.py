from .observer import Observer


class LRScheduler(Observer):
    """Abstract class to update learning rates.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate
            scheduler.

    """
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler


class BatchLRScheduler(LRScheduler):
    """Update learning rates at the end of each batch."""
    def update_on_batch_end(self):
        self.scheduler.step()
        print(self.scheduler.get_last_lr())


class EpochLRScheduler(LRScheduler):
    """Update learning rates at the end of each epoch."""
    def update_on_epoch_end(self):
        self.scheduler.step()


class ValidationLRScheduler(LRScheduler):
    """Update learning rates at the end of each epoch according to validation.

    Note:
        Override :meth:`get_loss` to return the validiation loss.

    """
    def get_loss(self):
        return self.subject.losses['model'].current.mean()

    def update_on_epoch_end(self):
        self.scheduler.step(self.get_loss())
        print(self.scheduler._last_lr)
