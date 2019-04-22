from disent.utils import getattr_with_default
from . import BaseHParamScheduler, register_hparam_scheduler


@register_hparam_scheduler('fixed')
class FixedSchedule(BaseHParamScheduler):
    def __init__(self, args, hparam, optimizer):
        super().__init__(args, hparam, optimizer)
        wu_attr = 'warmup_updates_{}'.format(hparam)
        self.warmup_updates = getattr(args, wu_attr, 0)
        setattr(args, wu_attr, self.warmup_updates)

        if self.warmup_updates > 0:
            self.warmup_factor = 1. / warmup_updates
        else:
            self.warmup_factor = 1
        self.step(0)

    @staticmethod
    def add_args(parser, hparam):
        wu_attr = '--warmup-updates-{}'.format(hparam)
        
        parser.add_argument(wu_attr, default=0, type=int, metavar='N',
                            help='warmup the hparam linearly for the first N updates')

    def get_next_value(self, epoch):
        values = getattr_with_default(self.args, self.hparam, [1.])
        return values[min(epoch, len(values) - 1)]

    def step(self, epoch, val_loss=None):
        super().step(epoch, val_loss)
        self.value = self.get_next_value(epoch)
        self.optimizer.set_hparam(self.hparam, self.warmup_factor * self.value)
        return self.optimizer.get_hparam(self.hparam)

    def step_update(self, num_updates):
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            self.optimizer.set_hparam(self.hparam, self.warmup_factor * self.value)
        return self.optimizer.get_hparam(self.hparam)
