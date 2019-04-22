import importlib
import os

from .base_task import BaseTask

TASK_REGISTRY = {}


def setup_task(args):
    return TASK_REGISTRY[args.task].setup_task(args)


def register_task(name):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError('Task ({}: {}) must extend BaseTask'.format(name, cls.__name__))
        
        TASK_REGISTRY[name] = cls

    return register_task_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]
        importlib.import_module('disent.tasks.' + task_name)
