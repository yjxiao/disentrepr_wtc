import torch

from disent import metrics, options, progress_bar, tasks, utils
from disent.meters import StopwatchMeter


def main(args):
    assert args.path is not None, '--path required for evaluation'
    print(args)
    use_cuda = torch.cuda.is_available() and not args.cpu

    task = tasks.setup_task(args)
    task.load_dataset()

    print('| loading model from {}'.format(args.path))
    model, model_args = utils.load_model_for_inference(
        args.path, task)
    if use_cuda:
        model.cuda()

    metric = metrics.build_metric(args)
    dummy_progress = progress_bar.build_progress_bar(args, [])
    
    eval_timer = StopwatchMeter()
    eval_timer.start()
    stats = metric.evaluate(task, model, args.seed)
    eval_timer.stop()

    print('| evaluation done in {:.1f}s'.format(eval_timer.sum))
    dummy_progress.print(stats)


def cli_main():
    parser = options.get_evaluation_parser()
    args = options.parse_args(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
