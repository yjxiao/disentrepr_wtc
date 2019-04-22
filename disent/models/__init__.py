import importlib
import os

from .base_model import BaseModel


MODEL_REGISTRY = {}


def build_model(args):
    return MODEL_REGISTRY[args.vae_arch].build_model(args)


def build_adversarial(args):
    return MODEL_REGISTRY[args.adversarial_arch].build_model(args)


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError('Model ({}: {}) must extend BaseFairseqModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        importlib.import_module('disent.models.' + module)
