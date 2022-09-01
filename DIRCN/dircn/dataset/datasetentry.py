import h5py

from pathlib import Path
from ..logger import get_logger


class DatasetEntry(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(self,
                 kspace_path: Path = None,
                 image_path: Path = None,
                 datasetname: str = None,
                 dataset_type: str = None):
        """
        Args:
"            kspace_path (Path): The path where the data is stored
            datasetname (str): The name of the dataset the data is from
            dataset_type (str): What kind of data the data is
        """

        self.logger = get_logger(name=__name__)
        self.kspace_path = str(kspace_path)
        self.image_path = str(image_path)
        self.datasetname = datasetname
        self.dataset_type = dataset_type

    def open(self):
        """
        Open the file
        Args:
        returns:
            the opened file
        """
        kspace = self.open_hdf5(self.kspace_path)
        image = self.open_hdf5(self.image_path)
        return kspace, image

    def open_hdf5(self, file):
        return h5py.File(file, 'r')