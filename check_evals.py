import os

result_dir = 'results'

template = 'M-{}.D-{}.T-{}.{}-{}.S-{}.csv'

datasets = ['dsprites', 'cars3d', 'shapes3d']
metrics = ['factor', 'mig', 'modularity', 'recon']
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


for task, hparam in zip(tasks, hparams):
    for dataset in datasets:
        for metric in metrics:
            for seed in seeds:
                if check(result_dir, task, hparam, dataset, metric, seed):
                    print('| found [{}] results on [{}] ({}) with seed {}'.format(
                        task, dataset, metric, seed))


