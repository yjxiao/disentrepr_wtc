import os
import csv

import torch

from disent import metrics, options, progress_bar, tasks, utils
from disent.meters import StopwatchMeter


def main(args):
    assert args.path is not None, '--path required for evaluation'
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

    metric = metrics.build_metric(args)

    pb = progress_bar.build_progress_bar(args, range(args.num_evals))
    results = []
    eval_timer = StopwatchMeter()
    eval_timer.start()
    for i in pb:
        seed = args.seed + i * 73    # different seed for each evaluation; 73 is arbitrary
        stats = metric.evaluate(task, model, seed)
        stats['seed'] = seed
        pb.print(stats)
        results.append(stats)
    eval_timer.stop()

    print('| evaluation done in {:.1f}s'.format(eval_timer.sum))
    if args.save_results:
        filename = get_identifier(args)
        filepath = os.path.join(args.save_dir, filename)
        fieldnames = stats.keys()
        with open(filepath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stats in results:
                writer.writerow(stats)
        print('| results saved to {}'.format(filepath))


def get_identifier(args):
    hp_str = ''
    segs = args.path.split('/')
    seed = segs[-2]    # assuming the folder directly containing the file is the seed number 
    for seg in segs:
        if seg.startswith('beta'):
            hp_str += '.{}'.format(seg.upper())
        elif seg.startswith('gamma'):
            hp_str += '.{}'.format(seg.upper())
            
    return 'M-{}.D-{}.T-{}{}.S-{}.csv'.format(
        args.metric, args.dataset, args.task, hp_str, seed)

    
def cli_main():
    parser = options.get_evaluation_parser()
    args = options.parse_args(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
