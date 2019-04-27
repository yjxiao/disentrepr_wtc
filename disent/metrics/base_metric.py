import torch


class BaseMetric(object):
    def __init__(self, args):
        self.args = args
        self.cuda = torch.cuda.is_available() and not args.cpu
        
    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, task, model, seed):
        raise NotImplementedError
