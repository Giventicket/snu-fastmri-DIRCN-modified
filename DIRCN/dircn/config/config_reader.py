import json
import torch


class ConfigReader(object):
    """
    class for reading the config file provided and make the optimizer and lr_scheduler
    and some small calls to make it easier to fetch things like the batch size etc.
    """

    def __init__(self, config: str):
        """
        Args:
            config (dict, str): The config dict or path to config json file.
        """
        with open(config, 'r') as inifile:
            self.config = json.load(inifile)

    def __getitem__(self, key):
        return self.config[key]

    def __str__(self):
        return self.config

    @property
    def shuffle(self):
        """
        Fetch shuffle in config dict
        """
        return self['data_loader']['args']['shuffle']

    @property
    def batch_size(self):
        """
        Fetch batch_size in config dict
        """
        return self['data_loader']['args']['batch_size']

    @property
    def num_workers(self):
        """
        Fetch num_workers in config dict
        """
        return self['data_loader']['args']['num_workers']

    def optimizer(self, model_params):
        """
        Create optimizer:
        Args:
            model_params: The paramters for the weights the optimizer is for
        returns:
            optimizer: The optmizer for the given weights params
        """
        optim = self.config['optimizer']
        optimizer = getattr(torch.optim, optim['type'])
        optimizer = optimizer(model_params,
                              lr=optim['args']['lr'],
                              weight_decay=optim['args']['weight_decay'],
                              amsgrad=optim['args']['amsgrad'],
                              betas=(optim['args']['beta1'], optim['args']['beta2']))
        return optimizer

    def lr_scheduler(self, optimizer):
        """
        Create learning rate scheduler:
        Args:
            optimizer (torch.optim): optimizer the lr_scheduler is supposed to manipulate
        returns:
            lr_scheduler: for the optimizer given
        """
        lr_sched = self.config['lr_scheduler']
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_sched['type'])
        lr_scheduler = lr_scheduler(optimizer=optimizer,
                                    milestones=lr_sched['args']['milestones'],
                                    gamma=lr_sched['args']['gamma'])
        return lr_scheduler

    def configs(self):
        """
        returns:
            the config themselvs
        """
        return self.config


