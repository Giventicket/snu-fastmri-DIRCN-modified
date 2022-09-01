import os
from typing import Callable, Dict, Union
from collections import defaultdict

import numpy as np

import torch
from ..base import BaseTrainer
from ..config import ConfigReader
from ..models import MultiMetric, MultiLoss
from ..preprocessing.crop_image import CropImage

from skimage.metrics import structural_similarity

from tqdm import tqdm
import h5py

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Union[MultiLoss, Callable],
                 metric_ftns: Union[MultiMetric, Dict[str, Callable]],
                 config: ConfigReader,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 test_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 log_step: int = None,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         config=config,
                         seed=seed,
                         device=device)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.len_epoch = len(data_loader)
        self.batch_size = data_loader.batch_size
        self.log_step = int(self.len_epoch / (4 * self.batch_size)) if not isinstance(log_step, int) else log_step
        self.crop = CropImage((384, 384))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        losses = defaultdict(list)
        pbar = tqdm(enumerate(self.data_loader), desc='Epoch ' + str(epoch))

        for batch_idx, (raw, data, mask, target, _) in pbar:
            data = data.to(self.device)
            mask = mask.type(torch.bool).to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()  # Clear before inputting to weights
            output = self.model(data, mask)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                gtSSIM = structural_similarity(
                    self.crop(target.to("cpu")).detach().numpy()[0],
                    self.crop(raw.to("cpu")).detach().numpy()[0], data_range=self.crop(target.to("cpu")).detach().numpy().max()
                )
                predSSIM = structural_similarity(
                    self.crop(target.to("cpu")).detach().numpy()[0],
                    self.crop(output.to("cpu")).detach().numpy()[0], data_range=self.crop(target.to("cpu")).detach().numpy().max()
                )

                pbar.set_description('Train {}: {} {} Loss: {:.6f} predSSIM: {:.6f} gtSSIM: {:.6f}'.format(
                    'Epoch',
                    epoch,
                    self._progress(batch_idx),
                    loss, 1 - predSSIM, 1 - gtSSIM))

        losses['loss_func'] = str(self.loss_function)
        losses['loss'] = [np.mean(losses['loss'])]

        return losses

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (raw, data, mask, target, property) in tqdm(enumerate(self.valid_data_loader),
                                                             desc='Epoch ' + str(epoch)):
                data = data.to(self.device)
                mask = mask.type(torch.bool).to(self.device)
                target = target.to(self.device)

                output = self.model(data, mask)

                loss = self.loss_function(self.crop(output), self.crop(target))

                metrics['loss'].append(loss.item())

                for key, metric in self.metric_ftns.items():
                    metrics[key].append(metric(
                        self.crop(output),
                        self.crop(target)
                    ).item())

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        return metric_dict

    def _test_epoch(self, epoch):
        """
        test after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about test
        """
        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (raw, data, mask, target, property) in tqdm(enumerate(self.test_data_loader),
                                                             desc='Epoch ' + str(epoch)):
                data = data.to(self.device)
                mask = mask.type(torch.bool).to(self.device)
                target = target.to(self.device)

                output = self.model(data, mask)

                loss = self.loss_function(self.crop(output), self.crop(target))

                metrics['loss'].append(loss.item())

                for key, metric in self.metric_ftns.items():
                    metrics[key].append(metric(
                        self.crop(output),
                        self.crop(target)
                    ).item())

                file_name, slice_index, slice_len = property
                slice_index = slice_index.to('cpu').item()
                slice_len = slice_len.to('cpu').item()
                file_name = file_name[0]


                # make reconstruction files
                try:
                    f = h5py.File("../result/DIRCN/" + file_name, 'a')
                except:
                    os.mkdir("../result/DIRCN/")
                    f = h5py.File("../result/DIRCN/" + file_name, 'a')

                output = self.crop(output)
                try:
                    recons = np.array(f['reconstruction'])
                    f['reconstruction'][slice_index] = output[0].detach().to('cpu').numpy()
                except:
                    recons = np.zeros((slice_len, output.shape[1], output.shape[2]))
                    recons[slice_index] = output[0].detach().to('cpu').numpy()
                    f['reconstruction'] = recons

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        return metric_dict

    def _make_recons(self):
        """
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (_, data, mask, _, property) in tqdm(enumerate(self.test_data_loader)):
                file_name, slice_index, slice_len = property
                slice_index = slice_index.to('cpu').item()
                slice_len = slice_len.to('cpu').item()
                file_name = file_name[0]

                data = data.to(self.device)
                mask = mask.type(torch.bool).to(self.device)

                output = self.model(data, mask)
                output = self.crop(output)


                try:
                    f = h5py.File("../result/DIRCN/" + file_name, 'a')
                except:
                    os.mkdir("../result/DIRCN/")
                    f = h5py.File("../result/DIRCN/" + file_name, 'a')

                try:
                    recons = np.array(f['reconstruction'])
                    f['reconstruction'][slice_index] = output[0].detach().to('cpu').numpy()
                except:
                    recons = np.zeros((slice_len, output.shape[1], output.shape[2]))
                    recons[slice_index] = output[0].detach().to('cpu').numpy()
                    f['reconstruction'] = recons
        return

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.data_loader.batch_size
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
