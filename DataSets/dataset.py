import torch as t
import numpy as np
from torch.utils.data import Dataset,DataLoader


class DataSet(Dataset):
    def __init__(self):
        super(DataSet, self).__init__()
        self.dataframe = None #df


    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.dataframe.index)

    def get_dataloader(self):
        self.loader = None
        pass