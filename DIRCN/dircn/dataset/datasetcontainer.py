import contextlib
import random
from pathlib import Path
from typing import List

from tqdm import tqdm

from .datasetentry import DatasetEntry
from .datasetinfo import DatasetInfo
from ..logger import get_logger


@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class DatasetContainer(object):
    """
    Method to save and store the information about the dataset
    This includes the locations, etc.
    This means that all files can be stored at the same place or separately,
    it is cleaner to handle than just lumping files together in folders.

    And it is iterable and savable
    """

    def __init__(self,
                 info: List[DatasetInfo] = None,
                 entries: List[DatasetEntry] = None
                 ):
        """
        Args:
            info (list(DatasetInfo)): list of information about the dataset used
            entries (list(DatasetEntries)): list of the entries, i.e., the files of the dataset
        """

        self.logger = get_logger(name=__name__)
        self.info = info if info is not None else list()
        self.entries = entries if entries is not None else list()

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def shuffle(self, seed=None):
        """
        Shuffles the entries, used for random training
        Args:
            seed (int): The seed used for the random shuffle
        """
        with temp_seed(seed):
            random.shuffle(self.entries)

    def fastMRI(self,
                path_kspace: str,
                path_image: str,
                datasetname: str,
                dataset_type: str,
                source: str = 'fastMRI',
                dataset_description: str = 'Data for fastMRI challenge'):

        """
        Fills up the container using the folder containing the fastMRI data
        Args:
            path_kspace (str): Path to fastMRI data
            datasetname (str): Name of dataset
            dataset_type (str): The type of dataset this is
            source (str): The source of the dataset (fastMRI)
            dataset_description (str): description of dataset
        returns:
            DatasetContainer filled with fastMRI data
        """

        # ex path='input/train/kspace'
        root = Path.cwd().parent.parent
        path_kspace = root / Path(path_kspace)
        path_image = root / Path(path_image)

        files_kspace = list(path_kspace.glob('*.h5'))
        files_image = list(path_image.glob('*.h5'))


        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )

        self.info.append(info)

        for idx in range(len(files_kspace)):
            self.entries.append(DatasetEntry(datasetname=datasetname, dataset_type=dataset_type,))

        for idx, file in tqdm(enumerate(sorted(files_kspace))):
            self.entries[idx].kspace_path = file

        for idx, file in tqdm(enumerate(sorted(files_image))):
            self.entries[idx].image_path = file

        return self
