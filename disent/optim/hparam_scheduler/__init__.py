import importlib
import os

from .base_hparam_scheduler import BaseHParamScheduler

HPARAM_SCHEDULER_REGISTRY = {}


def build_hparam_scheduler(args, hparam, optimizer):
    return HPARAM_SCHEDULER_REGISTRY[args.hparam_scheduler](args, hparam, optimizer)


def register_hparam_scheduler(name):
    def register_hparam_scheduler_cls(cls):
        if name in HPARAM_SCHEDULER_REGISTRY:
            raise ValueError('Cannot register duplicate HParam scheduler ({})'.format(name))
        if not issubclass(cls, BaseHParamScheduler):
            raise ValueError('HParam Scheduler ({}: {}) must extend BaseHParamScheduler'.format(name, cls.__name__))
        HPARAM_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_hparam_scheduler_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('disent.optim.hparam_scheduler.' + module)
