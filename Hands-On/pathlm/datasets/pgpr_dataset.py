from pathlm.datasets.kg_dataset_base import KARSDataset

class PGPRDataset(KARSDataset):
    def __init__(self, args, set_name='train', data_dir=None):
        super().__init__(args, set_name, data_dir)