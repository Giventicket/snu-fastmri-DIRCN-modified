class DatasetInfo(object):
    """
    Information about the data
    """

    DATASET_TYPES = ['train', 'validation', 'test']  # Predefined types

    def __init__(self,
                 datasetname: str = None,
                 dataset_type: str = None,
                 source: str = None,
                 dataset_description: str = None):

        self._datasetname = datasetname
        self._dataset_type = dataset_type
        self._source = source
        self._dataset_description = dataset_description

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()