import torch
import torchvision
import numpy as np

from .datasetcontainer import DatasetContainer
from ..logger import get_logger

class DatasetLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 train_transforms: torchvision.transforms = None,
                 target_transforms: torchvision.transforms = None,
                 ):
        """
        Args:
            datasetcontainer: The datasetcontainer that is to be loaded
            train_transforms: Transforms the data is gone through before weights input
            target_transforms: Transforms the data is gone through before being ground truths
            img_key: potential key for opening the file
        """

        self.datasetcontainer = datasetcontainer
        self.train_transforms = train_transforms
        self.target_transforms = target_transforms

        self.logger = get_logger(name=__name__)

        # Create a dict that maps image index to file and image in file index
        self._kspaces = dict()
        counter = 0
        for i, entry in enumerate(datasetcontainer):
            kspaces, _ = entry.open()
            slice_len = kspaces['kspace'].shape[0]
            for j in range(slice_len):
                self._kspaces[counter] = (i, j, slice_len, str(entry.kspace_path).split('/')[-1])
                counter += 1
        self.logger.info('--------------------------------------------------------------')

    def __len__(self):
        return len(self._kspaces)

    def __getitem__(self, index):
        file_index, slice_index, slice_len, file_name = self._kspaces[index]

        entry = self.datasetcontainer[file_index]

        kspace_object, image_object = entry.open()

        kspace = kspace_object['kspace'][slice_index]
        mask = np.array(kspace_object['mask'])

        if entry.dataset_type == 'train':
            target = self.target_transforms(kspace)
        else: # val or test
            target = torch.tensor(image_object['image_label'][slice_index])

        kspace[:, :, mask != 1.0] = 0
        train = self.train_transforms(kspace)
        raw = self.target_transforms(kspace)

        mask = mask.reshape(1, 1, -1, 1)
        return raw, train, mask, target, (file_name, slice_index, slice_len)

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):
        if not self.current_index < self.max_length:
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item



