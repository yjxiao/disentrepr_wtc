from collections import OrderedDict

import torch

from disent import optim, utils
from disent.meters import AverageMeter, StopwatchMeter, TimeMeter
from disent.optim import lr_scheduler, hparam_scheduler


class Trainer(object):
    def __init__(self, args, task, model, criterion):
        self.args = args
        self.task = task
        self.criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self._model = self._model.cuda()
            self.criterion = self.criterion.cuda()

        self._lr_schedulers = None
        self._hp_schedulers = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizers = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()

        self.meters['ups'] = TimeMeter()
        self.meters['gnorm'] = AverageMeter()
        self.meters['clip'] = AverageMeter()
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds
        for comp in self.criterion.loss_components:
            self.meters['train_' + comp] = AverageMeter()
            self.meters['valid_' + comp] = AverageMeter()

    @property
    def model(self):
        return self._model

    @property
    def optimizers(self):
        if self._optimizers is None:
            self._build_optimizers()
        return self._optimizers

    def get_optimizer(self, name='main'):
        return self.optimizers[name]
    
    @property
    def lr_schedulers(self):
        if self._lr_schedulers is None:
            self._build_optimizers()
        return self._lr_schedulers

    def get_lr_scheduler(self, name='main'):
        return self.lr_schedulers[name]
    
    @property
    def hparam_schedulers(self):
        if self._hp_schedulers is None:
            self._build_optimizers()
        return self._hp_schedulers

    def get_hparam_scheduler(self, name):
        return self.hparam_schedulers[name]
    
    def _build_optimizers(self):
        self._optimizers = {}
        self._lr_schedulers = {}
        self._hp_schedulers = {}
        # this means the model includes adversarial task
        if isinstance(self.model, nn.ModuleDict):
            for key, model in self.model.items():
                params = list(filter(lambda p: p.requires_grad, model.parameters()))
                optimizer = optim.build_optimizer(self.args, params)
                self._optimizers[key] = optimizer
                self._lr_schedulers[key] = lr_scheduler.build_lr_scheduler(self.args, optimizer)
        else:
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            optimizer = optim.build_optimizer(self.args, params)
            self._optimizers['main'] = optimizer
            self._lr_schedulers['main'] = lr_scheduler.build_lr_scheduler(self.args, optimizer)

        # only build hparam scheduler for the main model
        for hparam in self.args.hparams:
            self._hp_schedulers[hparam] = hparam_scheduler.build_hparam_scheduler(
                self.args, hparam, self._optimizers['main'])

    def save_checkpoint(self, filename, extra_state):
        extra_state['train_meters'] = self.meters
        save_state(
            filename, self.args, self.model.state_dict(),
            self.optimizers, self.lr_schedulers, self.hparam_schedulers,
            self._num_updates, self._optim_history, extra_state)

    def load_checkpoint(self, filename, reset_optimizers=False,
                        reset_lr_schedulers=False,
                        optimizer_overrides=None):
        extra_state, self._optim_history, last_optim_state = load_model_state(
            filename, self.model)
        if last_optim_state is not None and not reset_optimizers:
            self._build_optimizers()
            
            last_optim = self._optim_history[-1]
            for key in self.optimizers:
                optimizer = self.get_optimizer(key)
                lr_scheduler = self.get_lr_scheduler(key)
                assert last_optim[key]['optimizer_name'] == optimizer.__class__.__name__, \
                    'optimizer does not match; please reset the optimizer (--reset-optimizer)'
                if not reset_lr_schedulers:
                    lr_scheduler.load_state_dict(last_optim[key]['lr_scheduler_state'])
                optimizer.load_state_dict(last_optim_state[key], optimizer_overrides)

                self._num_updates = last_optim['num_updates']

            for hparam in self.hparam_schedulers:
                scheduler = self.get_hparam_scheduler(hparam)
                scheduler.load_state_dict(last_optim['main'][hparam]['hparam_scheduler_state'])
                
        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state

    def train_step(self, sample):
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        self.meters['train_wall'].start()

        sample = self._prepare_sample(sample)

        loss, batch_size, logging_output = self.task.train_step(
            sample, self.model, self.criterion, self.get_optimizer())

        try:
            grad_norm = 0
            if not isinstance(loss, dict):
                loss = {'main': loss}
                
            k = len(loss)
            for i, (comp, comp_loss) in enumerate(loss.items()):
                optimizer = self.get_optimizer(comp)
                optimizer.zero_grad()
                retain_graph = i < k - 1
                loss.backward(retain_graph=retain_graph)
                grad_norm += optimizer.clip_grad_norm(self.args.clip_norm)
                optimizer.step()
                
            self._num_updates += 1

            for comp in self.lr_schedulers:
                self.get_lr_scheduler(comp).step_update(self._num_updates)
                
            for hparam in self.hparam_schedulers:
                self.get_hparam_scheduler(hparam).step_update(self._num_updates)

            # update meters
            self.meters['ups'].update(1.)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 \
                else 0.)

            for comp in self.criterion.loss_components:
                comp_loss = logging_output.get(loss_comp, 0)
                self.meters['train_' + comp].update(comp_loss, batch_size)

            self.meters['train_loss'].update(logging_output.get('loss', 0), batch_size)

        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None
            
        self.meters['train_wall'].stop()
        
        return logging_output

    def valid_step(self, sample):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            sample = self._prepare_sample(sample)

            loss, batch_size, logging_output = self.task.valid_step(
                sample, self.model, self.criterion)

        for comp in self.criterion.loss_components:
            comp_loss = logging_output.get(loss_comp, 0)
            self.meters['valid_' + comp].update(comp_loss, batch_size)

        self.meters['valid_loss'].update(logging_output.get('loss', 0), batch_size)
        return logging_output

    def zero_grad(self, key=None):
        if key:
            self.get_optimizer(key).zero_grad()
        else:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None, key='main'):
        return self.get_lr_scheduler(key).step(epoch, val_loss)

    def lr_step_update(self, num_updates, key='main'):
        return self.get_lr_scheduler(key).step_update(num_updates)

    def get_lr(self, key='main'):
        return self.get_optimizer(key).get_lr()

    def hparam_step(self, epoch, val_loss, hparam):
        return self.get_hparam_scheduler(hparam).step(epoch, val_loss)

    def hparam_step_update(self, num_updates, hparam):
        return self.get_hparam_scheduler(hparam).step_update(num_updates)

    def get_hparam(self, hparam):
        return self.get_optimizer('main').get_hparam(hparam)

    def get_meter(self, name):
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        if self.cuda:
            sample = utils.move_to_cuda(sample)
        return sample

    def _set_seed(self):
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)
    
