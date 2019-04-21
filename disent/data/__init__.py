import importlib
import os

from .vision_dataset import VisionDataset
from .iterators import CountingIterator, EpochBatchIterator


DATASET_REGISTRY = {}


def build_dataset(args, shuffle):
    return DATASET_REGISTRY[args.dataset].build_dataset(args, shuffle)


def register_dataset(name):
    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError('Cannot register duplicate dataset ({})'.format(name))
        if not issubclass(cls, VisionDataset):
            raise ValueError('Dataset ({}: {}) must extend VisionDataset'.format(name, cls.__name__))
        DATASET_REGISTRY[name] = cls
        return cls
    return register_dataset_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('disent.data.' + module)
