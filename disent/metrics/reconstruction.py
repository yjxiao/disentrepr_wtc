from collections import OrderedDict

import torch
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from disent.meters import AverageMeter
from . import BaseMetric, register_metric


@register_metric('recon')
class Reconstruction(BaseMetric):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--eval-batches', type=int, metavar='N',
                            help='number of evaluation batches')
        
    def evaluate(self, task, model, seed):
        model = _check_model(model)
        model.eval()
        stats = OrderedDict()

        itr = task.get_batch_iterator(
            dataset=task.dataset,
            batch_size=self.args.batch_size,
            num_batches=self.args.eval_batches,
            seed=seed).next_epoch_itr(shuffle=False)
        rec_meter = AverageMeter()
        
        for batch in itr:
            if self.cuda:
                batch = utils.move_to_cuda(batch)
            outputs = model(batch)
            batch_size = batch['batch_size']
            rec = torch.sum(- outputs['x'].log_prob(batch['image'])) / batch_size
            rec_meter.update(rec.item(), batch_size)
            
        stats['reconstruction'] = rec_meter.avg
        return stats

    
def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

