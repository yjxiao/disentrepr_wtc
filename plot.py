import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='results',
                    help='directory to the evaluation results')
parser.add_argument('--dataset', type=str, default='dsprites',
                    choices=['dsprites', 'cars3d', 'shapes3d'],
                    help='the dataset evaluated')
parser.add_argument('--metric', type=str,
                    choices=['factor', 'mig', 'modularity'],
                    help='metric to plot')
parser.add_argument('--tasks', type=lambda x: x.split(','),
                    help='list of tasks to include in the plot'
                    'if only a single task is included, plot by'
                    'hyperparameters.')
parser.add_argument('--plot-type', type=str, default='violin',
                    choices=['violin', 'box'],
                    help='type of the plots to generate.')
parser.add_argument('--model-seeds', type=lambda x: x.split(','),
                    default='42',
                    help='seeds to consider separated by commas')
parser.add_argument('--save-dir', type=str, default='figs',
                    help='folder to save plots')

args = parser.parse_args()

FNAME_TEMPLATE = 'M-{}.D-{}.T-{}.{}-{}.S-{}.csv'
TASKS = set(['vae', 'tc', 'factor', 'wae', 'wtc', 'wtc_wae'])
HPARAM = {
    'vae': 'BETA',
    'tc': 'BETA',
    'factor': 'GAMMA',
    'wae': 'BETA',
    'wtc': 'GAMMA',
    'wtc_wae': 'BETA',
}
VALUES = {
    'vae': (1, 4, 8, 16),
    'tc': (1, 4, 8, 16),
    'factor': (10, 20, 40, 80),
    'wae': (1, 4, 8, 16),
    'wtc': (1, 4, 8, 16),
    'wtc_wae': (1, 4, 8, 16),    
}
MAIN_METRIC = {
    'factor': 'eval_acc',
    'mig': 'avg_mig',
    'modularity': 'modularity',
}
METRIC_LIMITS = {
    'factor': (0.5, 1.0),
    'mig': (0, 0.4),
    'modularity': (0.6, 1.0),
}
COLORS = {
    'vae': '#e41a1c',
    'tc': '#377eb8',
    'factor': '#4daf4a',
    'wae': '#984ea3',
    'wtc': '#ff7f00',
    'wtc_wae': '#ffff33',
}


def main(args):
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    results = parse_evaluation_results(args)
    ylim = METRIC_LIMITS[args.metric]
    # plot per task by hparams
    for task in args.tasks:
        savepath = os.path.join(
            args.save_dir,
            'M-{}.D-{}.T-{}.{}.pdf'.format(
                args.metric, args.dataset,
                task, args.plot_type
            )
        )
        colors = [COLORS[task] for _ in results[task]]
        # this is HPARAM x EVALS
        plot_and_save(results[task], colors, savepath, args.plot_type)
        
    # plot by tasks
    # this should be TASK x EVALS
    results = aggregate_task_results(results)
    savepath = os.path.join(
        args.save_dir,
        'M-{}.D-{}.{}.pdf'.format(
            args.metric, args.dataset, args.plot_type)
    )
    colors = [COLORS[task] for task in results]
    plot_and_save(results, colors, savepath, args.plot_type)


def plot_and_save(results, colors, savepath, plot_type, ylim=None):
    """Note that results is a dictionary of lists. """
    xlabels, values = zip(*results.items())
    widths = 0.25
    positions = np.arange(1, len(values) + 1) * widths * 1.5
    xlim = (positions[0] - widths * 0.75, positions[-1] + widths * 0.75)
    plt.figure(figsize=(len(values), 5))
    if plot_type == 'violin':
        plot_violin(values, positions, widths, colors)
    elif plot_type == 'box':
        plot_box(values, positions, widths, colors)
    else:
        raise ValueError('Unsupported plot type: ' + plot_type)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(positions, xlabels)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def plot_box(values, positions, widths, colors):
    medianprops = dict(linewidth=1, color='k')
    parts = plt.boxplot(
        values, sym='', patch_artist=True,
        positions=positions, widths=widths,
        medianprops=medianprops)
    for pc, color in zip(parts['boxes'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    
def plot_violin(values, positions, widths, colors):
    parts = plt.violinplot(
        values, showextrema=False,
        positions=positions, widths=widths
    )
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(values, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(values, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    plt.scatter(positions, medians, marker='o', color='white', s=10, zorder=3)
    plt.vlines(positions, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(positions, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def aggregate_task_results(results):
    new_results = {}
    for task, value in results.items():
        new_results[task] = np.concatenate(list(value.values()))
    return new_results


def parse_evaluation_results(args):
    results = {}
    for task in args.tasks:
        assert task in TASKS, 'unrecognized task: ' + task
        results[task] = {}
        hparam = HPARAM[task]
        values = VALUES[task]
        for val in values:
            val_results = []
            for seed in args.model_seeds:
                fpath = os.path.join(
                    args.input_dir,
                    FNAME_TEMPLATE.format(
                        args.metric, args.dataset,
                        task, hparam, val, seed
                    ))
                val_results.extend(
                    parse_file(fpath, args.metric)
                )
            results[task][val] = val_results
    return results


def parse_file(fpath, metric):
    field = MAIN_METRIC[metric]
    results = []
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(float(row[field]))
    return results


if __name__ == '__main__':
    main(args)
