from .. import BaseOptimizer


class BaseLRScheduler(object):
    def __init__(self, args, optimizer):
        super().__init__()
        if not isinstance(optimizer, BaseOptimizer):
            raise ValueError('optimizer must be an instance of BaseOptimizer')
        self.args = args
        self.optimizer = optimizer
        self.best = None

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        pass

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()
