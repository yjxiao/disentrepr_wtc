import os
import numpy as np
import pickle
from colorama import Fore, Style

result_dir = 'results'
record_file = 'results/list.pkl'
template = 'M-{}.D-{}.T-{}.{}-{}.S-{}.csv'

datasets = ['dsprites', 'cars3d', 'shapes3d']
metrics = ['mig', 'factor', 'modularity', 'recon', 'wmig', 'wmod']
tasks = ['vae', 'tc', 'factor', 'wae', 'wtc', 'wtc_wae']
hparams = ['BETA', 'BETA', 'GAMMA', 'BETA', 'GAMMA', 'BETA']
values = {
    'vae': (1, 4, 8, 16),
    'tc': (1, 4, 8, 16),
    'factor': (10, 20, 40, 80),
    'wae': (1, 4, 8, 16),
    'wtc': (1, 4, 8, 16),
    'wtc_wae': (1, 4, 8, 16),
}
seeds = [1, 11, 42, 73, 89]


def check(result_dir, task, hparam, dataset, metric, seed):
    for val in values[task]:
        filename = template.format(
            metric, dataset, task, hparam, val, seed
        )
        if not os.path.isfile(os.path.join(result_dir, filename)):
            return False

    return True


def wrap_green(s):
    return Fore.GREEN + s + Fore.RESET


def wrap_yellow(s):
    return Fore.YELLOW + s + Fore.RESET


def wrap_bright(s):
    return Style.BRIGHT + s + Style.RESET_ALL


def print_table(checks, prev_checks):
    metric_header = ''.join([' {:^5} |'.format(metric[:5]) for metric in metrics])
    header = '|{:^10}| seed |{}'.format('', metric_header)
    line = '|' + '-' * 65 + '|'
    rows = [header, line]
    for dataset in checks:
        prev_check = prev_checks.get(dataset, np.zeros((len(seeds), len(metrics))))
        for i, seed in enumerate(seeds):
            check_strs = ''
            for x, prev in zip(checks[dataset][i], prev_check[i]):
                if x > 0:
                    if prev > 0:
                        check_strs += '   ' + wrap_bright(wrap_green('✓')) + '   |'                        
                    else:
                        check_strs += '   ' + wrap_green('✓') + '   |'
                else:
                    check_strs += '   ' + wrap_yellow('☓') + '   |'
            row = '|{:^10}|{:^6}|{}'.format(dataset, seed, check_strs)
            rows.append(row)
        rows.append(line)
    print('\n'.join(rows))


if os.path.exists(record_file):
    with open(record_file, 'rb') as f:
        old_record = pickle.load(f)
else:
    old_record = {}
        
all_checks = {}
for task, hparam in zip(tasks, hparams):
    all_checks[task] = {}
    for dataset in datasets:
        all_checks[task][dataset] = np.zeros((len(seeds), len(metrics)))
        for i, metric in enumerate(metrics):
            for j, seed in enumerate(seeds):
                all_checks[task][dataset][j, i] = check(
                    result_dir, task, hparam, dataset, metric, seed)


for task in tasks:
    print('\n--{:-<65}'.format(task))
    print_table(all_checks[task], old_record.get(task, {}))

with open(record_file, 'wb') as f:
    pickle.dump(all_checks, f)
