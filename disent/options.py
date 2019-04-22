import argparse

import torch

from disent.models import MODEL_REGISTRY
from disent.optim import OPTIMIZER_REGISTRY
from disent.optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from disent.optim.hparam_scheduler import HPARAM_SCHEDULER_REGISTRY
from disent.tasks import TASK_REGISTRY


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename in save-dir from which to load checkpoint')
    group.add_argument('--reset-optimizers', action='store_true',
                       help='if set, does not load optimizer state from the checkpoint')
    group.add_argument('--reset-lr-schedulers', action='store_true',
                       help='if set, does not load lr scheduler state from the checkpoint')
    group.add_argument('--optimizer-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override optimizer args when loading a checkpoint')
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep the last N checkpoints saved with --save-interval-updates')
    group.add_argument('--keep-last-epochs', type=int, default=-1, metavar='N',
                       help='keep last N epoch checkpoints')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                       help='validate every N epochs')
    group.add_argument('--no-validate', action='store_true',
                       help='if set, does not validate')
    return group


def add_dataset_args(parser, train=False):
    group = parser.add_argument_group('Dataset and data loading')
    group.add_argument('--batch-size', type=int, default=64, metavar='N',
                       help='number of examples in a batch')
    return group


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')
    group.add_argument('--vae-arch', default='conv_vae', metavar='ARCH',
                       help='VAE model architecture')
    group.add_argument('--adversarial-arch', default='mlp_discriminator', metavar='ARCH',
                       help='adversarial model architecture')
    return group

    
def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--max-epoch', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--clip-norm', default=0, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--optimizer', default='adam', metavar='OPT',
                       choices=OPTIMIZER_REGISTRY.keys(),
                       help='Optimizer')
    group.add_argument('--lr', '--learning-rate', default='0.0001', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--momentum', default=0.99, type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')
    group.add_argument('--lr-scheduler', default='fixed',
                       choices=LR_SCHEDULER_REGISTRY.keys(),
                       help='Learning Rate Scheduler')
    group.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                       help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
    group.add_argument('--min-lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--hparam-scheduler', default='fixed',
                       choices=HPARAM_SCHEDULER_REGISTRY.keys(),
                       help='hyper parameter scheduler (used for kld annealing, etc.)')
    return group


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def get_training_parser(default_task='vae'):
    parser = get_parser('Trainer', default_task)
    add_dataset_args(parser, train=True)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_parser(desc, default_task):
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', default=None, help='log format to use',
                        choices=['json', 'none', 'simple', 'tqdm'])
    parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
                        help='path to save logs for tensorboard, should match --logdir '
                             'of running tensorboard (default: no tensorboard logging)')
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--device-id', default=0, type=int,
                        help='gpu id to use')
    parser.add_argument('--task', metavar='TASK', default=default_task,
                        choices=TASK_REGISTRY.keys(),
                        help='task')
    return parser


def parse_args(parser, input_args=None, parse_known=False):
    args, _ = parser.parse_known_args(input_args)

    if hasattr(args, 'vae_arch'):
        MODEL_REGISTRY[args.vae_arch].add_args(parser)
    if hasattr(args, 'adversarial_arch'):
        MODEL_REGISTRY[args.adversarial_arch].add_args(parser)
    if hasattr(args, 'task'):
        TASK_REGISTRY[args.task].add_args(parser)
    if hasattr(args, 'optimizer'):
        OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    if hasattr(args, 'lr_scheduler'):
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(parser)
    if hasattr(args, 'hparam_scheduler'):
        for hparam in TASK_REGISTRY[args.task].hparams:
            HPARAM_SCHEDULER_REGISTRY[args.hparam_scheduler].add_args(parser, hparam)
    if parse_known:
        return parser.parse_known_args(input_args)
    else:
        return parser.parse_args(input_args)
