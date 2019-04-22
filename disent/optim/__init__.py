import importlib
import os

from .base_optimizer import BaseOptimizer

OPTIMIZER_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


def build_optimizer(args, params):
    params = list(filter(lambda p: p.requires_grad, params))
    return OPTIMIZER_REGISTRY[args.optimizer](args, params)


def register_optimizer(name):
    """Decorator to register a new optimizer."""

    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        if not issubclass(cls, BaseOptimizer):
            raise ValueError('Optimizer ({}: {}) must extend BaseOptimizer'.format(name, cls.__name__))
        if cls.__name__ in OPTIMIZER_CLASS_NAMES:
            # We use the optimizer class name as a unique identifier in
            # checkpoints, so all optimizer must have unique class names.
            raise ValueError('Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
        OPTIMIZER_REGISTRY[name] = cls
        OPTIMIZER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_optimizer_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('disent.optim.' + module)
