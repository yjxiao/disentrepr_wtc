from collections import OrderedDict
import os
import re

import torch
from torch.serialization import default_restore_location


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def getattr_with_default(args, attr, default):
    value = getattr(args, attr, default)
    if value is None:
        value = default

    return value


def load_model_for_inference(filename, task):
    if not os.path.exists(filename):
        raise IOError('Model file not found: {}'.format(filename))
    
    state = torch.load(
        filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = state['args']
    model = task.build_model(args)
    model.load_state_dict(state['model'], strict=True)
    return model, args


def load_model_state(filename, model):
    if not os.path.exists(filename):
        return None, None, None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    try:
        model.load_state_dict(state['model'], strict=True)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')
    
    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']

    
def save_state(filename, args, model_state_dict, optimizers,
               lr_schedulers, hp_schedulers, num_updates,
               optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}

    optim_info = {'num_updates': num_updates}
    optim_state = {}
    for key in optimizers:
        optimizer = optimizers[key]
        lr_scheduler = lr_schedulers[key]
        optim_info[key] = {
            'optimizer_name': optimizer.__class__.__name__,
            'lr_scheduler_state': lr_scheduler.state_dict(),
        }
        optim_state[key] = convert_state_dict_type(optimizer.state_dict())
        if key == 'main':
            for hparam, hp_scheduler in hp_schedulers.items():
                optim_info[key][hparam] = {
                    'hparam_scheduler_state': hp_scheduler.state_dict()
                }

    state_dict = {
        'args': args,
        'model': model_state_dict if model_state_dict else {},
        'optimizer_history': optim_history + [optim_info],
        'last_optimizer_state': optim_state,
        'extra_state': extra_state,
    }
    torch.save(state_dict, filename)


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)
