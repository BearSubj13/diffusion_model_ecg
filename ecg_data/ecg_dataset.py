import numpy as np
import torch
from torch.utils.data import Dataset


class ECGdataset(Dataset):
    def __init__(self, dataset_path, terminate=False):
        super(ECGdataset, self).__init__()
        self.ecg_tensor = np.load(dataset_path)
        self.ecg_tensor = torch.FloatTensor(self.ecg_tensor)

    def __getitem__(self, index):
        if self.ecg_tensor.dim() == 2:
            return self.ecg_tensor[index, :] 
        else:
            x = self.ecg_tensor[index % 10, :, :]
            x = x[[0,2],:]
            return x 

    def __len__(self):
        dataset_size = self.ecg_tensor.shape[0]
        return dataset_size
