import importlib
import os

from .base_metric import BaseMetric


METRIC_REGISTRY = {}


def build_metric(args):
    return METRIC_REGISTRY[args.metric](args)


def register_metric(name):
    def register_metric_cls(cls):
        if name in METRIC_REGISTRY:
            raise ValueError('Cannot register duplicate metric ({})'.format(name))
        if not issubclass(cls, BaseMetric):
            raise ValueError('Metric ({}: {}) must extend BaseMetric'.format(name, cls.__name__))
        METRIC_REGISTRY[name] = cls
        return cls

    return register_metric_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('disent.metrics.' + module)
