class BaseMetric(object):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def add_args(parser):
        pass

    def evaluate(self, task, model, seed):
        raise NotImplementedError
