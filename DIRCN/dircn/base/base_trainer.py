import time
import sys

from typing import Union, Dict
from abc import abstractmethod
from pathlib import Path
from datetime import datetime

import torch

import numpy as np

from ..logger import get_logger
from ..models import MultiLoss, MultiMetric
from ..metrics import MetricTracker
from ..config import ConfigReader


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: MultiLoss,
                 metric_ftns: Union[MultiMetric, Dict[str, callable]],
                 config: ConfigReader,
                 seed: int = None,
                 device: str = None,
                 ):
        """
        Args:
            model (torch.nn.Module): The weights to be trained
            loss_function (MultiLoss): The loss function or loss function class
            metric_ftns (MultiMetric, Dict[str, callable]): Dict or Multimetric for the metrics to be evaluated during validation
            config (dict): dict of configs
            seed (int): integer seed to enforce non stochasticity,
            device (str): string of the device to be trained on, e.g., "cuda:0"
        """

        # Reproducibility is a good thing
        if isinstance(seed, int):
            torch.manual_seed(seed)

        self.config = config
        self.logger = get_logger(name=__name__)

        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = config.optimizer(model_params=self.model.parameters())
        self.lr_scheduler = config.lr_scheduler(optimizer=self.optimizer)
        self.loss_function = loss_function.to(self.device)

        if isinstance(metric_ftns, dict):  # dicts can't be sent to the gpu
            self.metrics_is_dict = True
            self.metric_ftns = metric_ftns
        else:  # MetricTracker class can be sent to the gpu
            self.metrics_is_dict = False
            self.metric_ftns = metric_ftns.to(self.device)

        trainer_cfg = config.configs()['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']

        self.iterative = bool(trainer_cfg['iterative'])
        self.iterations = int(trainer_cfg['iterations'])

        self.start_epoch = 1

        self.checkpoint_dir = Path(trainer_cfg['save_dir']) / Path(datetime.today().strftime('%Y-%m-%d'))

        self.metric = MetricTracker(config=config.configs())

        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic after an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self, epoch):
        """
        test logic after an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError


    def train(self):
        """
        Full training logic
        """
        # Use iterations or epochs
        epochs = self.iterations if self.iterative else self.epochs

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start_time = time.time()
            loss_dict = self._train_epoch(epoch)
            val_dict = self._valid_epoch(epoch)
            test_dict = self._test_epoch(epoch)
            epoch_end_time = time.time() - epoch_start_time

            val_loss = np.mean(np.array(val_dict['loss']))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save logged information regarding this epoch/iteration
            self.metric.training_update(loss=loss_dict, epoch=epoch)
            self.metric.validation_update(metrics=val_dict, epoch=epoch)

            self.logger.info('Epoch/iteration {} with validation completed in {}, '\
                'run mean statistics:'.format(epoch, epoch_end_time))

            if hasattr(self.lr_scheduler, 'get_last_lr'):
                self.logger.info('Current learning rate: {}'.format(self.lr_scheduler.get_last_lr()))
            elif hasattr(self.lr_scheduler, 'get_lr'):
                self.logger.info('Current learning rate: {}'.format(self.lr_scheduler.get_lr()))

            # print logged informations to the screen
            # training loss
            loss = np.array(loss_dict['loss'])
            self.logger.info('Mean training loss: {}'.format(np.mean(loss)))

            if val_dict is not None:
                for key, valid in val_dict.items():
                    valid = np.array(valid)
                    self.logger.info('Mean validation {}: {}'.format(str(key), np.mean(valid)))

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)
            if val_loss < self.min_validation_loss:
                self.min_validation_loss = val_loss
                self.save_checkpoint(epoch, best=True)

            self.logger.info('-----------------------------------')
        self.save_checkpoint(epoch, best=False)
        self.metric.write_to_file(path=self.checkpoint_dir / Path('statistics.json'))  # Save metrics at the end

    def save_checkpoint(self, epoch, best: bool = False):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
            best (bool), save as best epoch so far, different naming convention
        """
        arch = type(self.model).__name__
        if self.lr_scheduler is not None:  # In case of None
            scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            scheduler_state_dict = None

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_state_dict,
            'config': self.config.configs(),
            'loss_func': str(self.loss_function),
            }

        if best:  # Save best case with different naming convention
            save_path = Path(self.checkpoint_dir) / Path('best_validation')
            filename = str(save_path / 'checkpoint-best.pth')
        else:
            save_path = Path(self.checkpoint_dir) / Path('epoch_' + str(epoch))
            filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))

        save_path.mkdir(parents=True, exist_ok=True)

        statics_save_path = save_path / Path('statistics.json')

        self.metric.write_to_file(path=statics_save_path)  # Save for every checkpoint in case of crash
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self,
                          resume_model: str,
                          resume_metric: str):
        """
        Resume from saved checkpoints
        Args:
            resume_model (str): Checkpoint path, either absolute or relative
            resume_metric (str): Metric path, either absolute or relative
        """
        if not Path(resume_model).is_file():
            self.logger.warning('resume_model object does not exist, ensure that {} is correct, '
                                'aborting previous checkpoint loading'.format(str(resume_model)))
            return None

        resume_model = str(resume_model)
        self.logger.info("Loading checkpoint: {} ...".format(resume_model))

        checkpoint = torch.load(resume_model, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Different architecture from that given in the config, '
                                'this may yield and exception while state_dict is loaded.')

        freeze = []
        for k, v in self.model.state_dict().items():
            new_k = k.split(".")
            if new_k[0] == "i_cascades":
                pass
            else:
                freeze.append(k)

        for name, param in self.model.named_parameters():
            if name in freeze:
                param.requires_grad = False

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model = self.model.to(self.device)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Different optimizer from that given in the config, '
                                'optimizer parameters are not resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load lr_scheduler state from checkpoint only when lr_scheduler type is not changed.
        if checkpoint['config']['lr_scheduler']['type'] != self.config['lr_scheduler']['type']:
            self.logger.warning('Warning: Different scheduler from that given in the config, '
                                 'scheduler parameters are not resumed.')
        elif self.lr_scheduler is None:
            self.logger.warning('Warning: lr_scheduler is None, '
                                'scheduler parameters are not resumed.')
        else:
            if checkpoint['scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                self.logger.warning('Warning: lr_scheduler is saved as None, '
                                    'scheduler parameters cannot be resumed.')

        if resume_metric is None:
            self.logger.info('No path were given for prior statistics, cannot resume.')
            self.logger.info('New statistics will be written, and saved as regular.')
        else:
            self.metric.resume(resume_path=resume_metric)

        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

        self.checkpoint_dir = Path(resume_metric).parent.parent  # Ensuring the same main folder after resuming

        for key, value in self.metric[self.metric.VALIDATION_KEY].items():
            loss = np.mean(np.array(value['loss']))
            self.min_validation_loss = min(self.min_validation_loss, loss)
