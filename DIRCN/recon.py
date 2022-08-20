import torch

import torchvision
import numpy as np

from dircn import DatasetContainer
from dircn import DatasetLoader

from dircn.models import MultiLoss, MultiMetric

from dircn.models.dircn.dircn import DIRCN


from dircn.trainer import Trainer
from dircn.config import ConfigReader

from dircn.models.losses import GaussianSSIMLoss, SSIMLoss

from dircn.preprocessing import (
    KspaceToImage,
    ComplexNumpyToTensor,
    ComplexAbsolute,
    RSS,
    DownsampleFOV,
    )

train = DatasetContainer()
train.fastMRI(
    path_kspace='input/train/kspace',
    path_image='input/train/image',
    datasetname='fastMRI',
    dataset_type='train'
)

valid = DatasetContainer()
valid.fastMRI(
    path_kspace='input/val/kspace',
    path_image='input/val/image',
    datasetname='fastMRI',
    dataset_type='val')

test = DatasetContainer()
test.fastMRI(
    path_kspace='input/leaderboard/kspace',
    path_image='input/leaderboard/image',
    datasetname='fastMRI',
    dataset_type='val')

# On the fly processing
train_transforms = torchvision.transforms.Compose([
    DownsampleFOV(k_size=384, i_size=384),
    lambda x: x.astype(np.complex64),
    ComplexNumpyToTensor(complex_support=False),
    ])


target_transforms = torchvision.transforms.Compose([
    DownsampleFOV(k_size=384, i_size=384),
    KspaceToImage(norm='ortho'),
    ComplexAbsolute(),  # Better to use numpy complex absolute than fastmri complex absolute
    lambda x: x.astype(np.float32),
    ComplexNumpyToTensor(complex_support=True),
    RSS(),
    ])

training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=train_transforms,
    target_transforms=target_transforms,
    )

validation_loader = DatasetLoader(
    datasetcontainer=valid,
    train_transforms=train_transforms,
    target_transforms=target_transforms
    )

test_loader = DatasetLoader(
    datasetcontainer=test,
    train_transforms=train_transforms,
    target_transforms=target_transforms
    )

# Loss function
loss = [(1, GaussianSSIMLoss()), (1, torch.nn.L1Loss())]
loss = MultiLoss(losses=loss)


# Histogram, skille for vev
# Radiomics for skille finstruktur, pyradiomics
metrics = {
    #'GaussianSSIMLoss': GaussianSSIMLoss(),
    'SSIMLoss': SSIMLoss(),
    'MSE': torch.nn.MSELoss(),
    'L1Loss': torch.nn.L1Loss()
}

metrics = MultiMetric(metrics=metrics)

# do under 9 million params
model = DIRCN(
    num_cascades=8,
    n=16,
    sense_n=4,
    groups=4,
    sense_groups=2,
    bias=True,
    ratio=1. / 8,
    dense=True,
    variational=False,
    interconnections=True,
    )

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

path = './dircn.json'
config = ConfigReader(config=path)
train_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle)


valid_loader = torch.utils.data.DataLoader(dataset=validation_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=False)

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    test_data_loader=test_loader,
    seed=None,
    log_step=1,
    device='cuda:0',
    )


trainer.resume_checkpoint(
    resume_model="/root/fastMRI/DIRCN/weights/best_validation/checkpoint-best.pth",
    resume_metric="/root/fastMRI/DIRCN/weights/best_validation/statistics.json"
)


trainer._make_recons() # reconstruct images of (384, 384)