import os

import numpy as np
import torch

from disent import options, progress_bar, tasks, utils
from disent.meters import StopwatchMeter


def main(args):
    assert args.path is not None, '--path required for generation'
    print(args)
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    task = tasks.setup_task(args)
    task.load_dataset()

    print('| loading model from {}'.format(args.path))
    model, model_args = utils.load_model_for_inference(
        args.path, task)
    if use_cuda:
        model.cuda()

    itr = task.get_batch_iterator(
        dataset=task.dataset,
        batch_size=args.batch_size,
        num_batches=args.gen_batches,
        seed=args.seed+1
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)
    
    pb = progress_bar.build_progress_bar(args, itr)
    mod_values = get_modification_values(args)
    for batch in pb:
        batch = utils.move_to_cuda(batch) if use_cuda else batch
        batch_size = batch['batch_size']
        
        if args.gen_mode == 'standard' or args.gen_mode is None:
            dims = []
        elif args.modify_dims is None:
            dims = range(model_args.code_size)
        else:
            dims = args.modify_dims
        mods = [{'dim': idx, 'values': mod_values} for idx in dims]
                
        gen_timer.start()
        batch_images = task.gen_step(generator, model, batch, mods)
        
        if len(mods) == 0:
            # (1, B)
            num_images = batch_size
            args.save_format = 'jpg'
            save_batch_images(batch['id'], batch_images[0], args)
        else:
            # (D, V, B)
            num_images = batch_size * len(mod_values) * len(mods)
            for i, dim in enumerate(dims):
                if args.save_format.lower() == 'gif':
                    for j in range(batch_size):
                        save_gifs(
                            batch['id'][j], batch_images[i, :, j],
                            args, dim=dim)
                else:
                    for j, val in enumerate(mod_values):
                        save_batch_images(
                            batch['id'], batch_images[i][j], args,
                            dim=dim, mod_val=val)
    
        gen_timer.stop(num_images)

    print('| generated {} images in {:.1f}s ({:.2f} images/s)'.format(
        gen_timer.n, gen_timer.sum, gen_timer.n / gen_timer.sum))


def save_batch_images(indices, images, args, dim=None, mod_val=None):
    for idx, image in zip(indices, images):
        savedir, filename = get_identifier(idx, args, dim=dim, mod_val=mod_val)
        savepath = os.path.join(savedir, filename)
        os.makedirs(savedir, exist_ok=True)
        image.save(savepath, format=args.save_format)

        
def save_gifs(idx, images, args, dim=None):
    savedir, filename = get_identifier(idx, args, dim=dim)
    savepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)    
    images[0].save(savepath, format='GIF', append_images=images[1:],
                   save_all=True, duration=100, loop=0)


def get_modification_values(args):
    if args.gen_mode == 'standard' or args.gen_mode is None:
        return None
    elif args.gen_mode == 'traversal':
        assert hasattr(args, 'traversal_range') and args.traversal_range is not None, \
            '--traversal-range is needed to perform traversal'
        assert len(args.traversal_range) == 2, '--traversal-range expects 2 values: MIN,MAX'
        assert hasattr(args, 'traversal_step') and args.traversal_step is not None, \
            '--traversal-step is needed to perform traversal'
        return np.arange(*args.traversal_range, args.traversal_step)
    else:
        raise ValueError("Unsupported generation mode " + args.gen_mode)

    
def get_identifier(idx, args, dim=None, mod_val=None):
    # -5: dataset; -4: task; -3: hparam; -2: seed
    basedir = '/'.join(args.path.strip().split('/')[-5:-1])
    savedir = os.path.join(args.save_dir, basedir, args.save_format)
    filename = "ID-{}{}{}.{}".format(
        idx, '.D-{:d}'.format(dim) if dim is not None else '',
        '.V-{:.1f}'.format(mod_val) if mod_val is not None else '',
        args.save_format)
    return savedir, filename

    
def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
