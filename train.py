from collections import OrderedDict
import os
import random
import math

import torch

from disent import options, progress_bar, tasks, utils
from disent.meters import StopwatchMeter
from disent.trainer import Trainer


def main(args):
    print(args)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    task = tasks.setup_task(args)
    task.load_dataset()
    args.hparams = task.hparams
    args.in_channels = task.dataset.in_channels
    
    # build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # build trainer
    trainer = Trainer(args, task, model, criterion)

    epoch_iter = task.get_batch_iterator(
        dataset=task.dataset,
        batch_size=args.batch_size,
        seed=args.seed)

    if not load_checkpoint(args, trainer, epoch_iter):
        print('| initializing new training session')

    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_loss = None

    while lr > args.min_lr and epoch_iter.epoch < max_epoch and trainer.get_num_updates() < max_update:
        train(args, trainer, task, epoch_iter)
        
        if epoch_iter.epoch % args.validate_interval == 0 and not args.no_validate:
            valid_loss = validate(args, trainer, task, epoch_iter, 'valid')

        lr = trainer.lr_step(epoch_iter.epoch, valid_loss)

        if epoch_iter.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_iter, valid_loss)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_iter):
    itr = epoch_iter.next_epoch_itr()
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_iter.epoch, no_progress_bar='simple')
    
    max_update = args.max_update or math.inf
    for i, sample in enumerate(progress, start=epoch_iter.iterations_in_epoch):
        log_output = trainer.train_step(sample)
        if log_output is None:
            continue

        stats = get_training_stats(trainer)
        progress.log(stats)
        num_updates = trainer.get_num_updates()
        if num_updates >= max_update:
            break

    stats = get_training_stats(trainer)
    stats['bsz'] = args.batch_size
    progress.print(stats)

    for name, meter in trainer.meters.items():
        if name in ['wall', 'train_wall'] or 'valid' in name:
            continue
        else:
            meter.reset()


def get_training_stats(trainer):
    stats = OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    stats['ups'] = trainer.get_meter('ups')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    for hparam in trainer.task.hparams:
        stats[hparam] = trainer.get_hparam(hparam)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_iter, split):
    itr = task.get_batch_iterator(
        dataset=task.dataset,
        batch_size=args.batch_size,
        seed=args.seed).next_epoch_itr(shuffle=False)

    progress = progress_bar.build_progress_bar(
        args, itr, epoch_iter.epoch,
        prefix='valid on {}'.format(split),
        no_progress_bar='simple')

    # reset validation meters
    for name, meter in trainer.meters.items():
        if 'valid' in name:
            meter.reset()
            
    for sample in progress:
        log_output = trainer.valid_step(sample)

    stats = get_valid_stats(trainer)
    stats['bsz'] = args.batch_size
    progress.print(stats)
    return stats['valid_loss']


def get_valid_stats(trainer):
    stats = OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    stats['num_updates'] = trainer.get_num_updates()

    if hasattr(save_checkpoint, 'best'):
        stats['best_loss'] = min(save_checkpoint.best, stats['loss'].avg)
    return stats


def save_checkpoint(args, trainer, epoch_iter, val_loss):
    if args.no_save:
        return

    epoch = epoch_iter.epoch
    end_of_epoch = epoch_iter.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_iter.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

        print('| saved checkpoint {} (epoch {} @ {} updates)'.format(
            checkpoints[0], epoch, updates))

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint(\d+)\.pt')
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_iter):
    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(
            checkpoint_path, args.reset_optimizers, args.reset_lr_schedulers,
            eval(args.optimizer_overrides))

        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_iter.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_iter.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_iter.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            trainer.hparam_step(epoch_iter.epoch)
            trainer.hparam_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    else:
        print('| no existing checkpoint found {}'.format(checkpoint_path))
    return False


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
