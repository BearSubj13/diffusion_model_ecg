#%%
import os
import numpy as np
import torch

from opendatasets import load_wsdb
from torch.utils.data import Dataset
from processing import preprocess

class ECGdataset(Dataset):
    def __init__(self, dataset_path, terminate=False):
        super(ECGdataset, self).__init__()
        self.ecg_tensor = np.load(dataset_path)
        self.ecg_tensor = torch.FloatTensor(self.ecg_tensor)

    def __getitem__(self, index):
        if self.ecg_tensor.dim == 2:
            return self.ecg_tensor[index % 400, :] 
        else:
            return self.ecg_tensor[index, :, :] 

    def __len__(self):
        dataset_size = self.ecg_tensor.shape[0]
        return dataset_size

#processing params
params = dict()
params['WINDOW'] = 8 #sec
params['CROP_METHOD'] = 'random' #during training, always "center" during test, valid and inference
params['SR'] = 128#64 - 256 points
params['HIGH_PASS'] = None
params['LOW_PASS'] = 40
params['SUBTRACT_RUNNING_MEAN'] = 63   


def ecg_preprocessing(folder_path, save_path, terminate=False):
    ecg_list = []
    for file_name in os.listdir(folder_path):
        if file_name[-3:] == "mat":
            file_path = os.path.join(folder_path, file_name)
            ecg, _, sr = load_wsdb(file_path)
            ecg = preprocess(ecg, sr, params)
            ecg_list.append(ecg[:,:4*params['SR']])
            ecg_list.append(ecg[:,2*params['SR']:6*params['SR']])
            ecg_list.append(ecg[:,4*params['SR']:8*params['SR']])
            ecg_list.append(ecg[:,6*params['SR']:10*params['SR']])
            ecg_list.append(ecg[:,-4*params['SR']:])
        if terminate:
            if len(ecg_list) > terminate:
                break
    
    #tensor_for_save = np.vstack(ecg_list)
    tensor_for_save = np.stack(ecg_list, axis=0)
    np.save(save_path, tensor_for_save)
    return ecg_list


if __name__ == "__main__":
    folder_path = "/ayb/vol1/kruzhilov/datasets/ecg/WFDB_Ningbo"
    save_path = "/ayb/vol1/kruzhilov/datasets/ecg/ningbo_128.npy"
    ecg_unit = ecg_preprocessing(folder_path, save_path, terminate=False)[3]
    # ecg_dataset = ECGdataset(save_path)
    # ecg_unit = ecg_dataset.__getitem__(30)
    # import matplotlib.pyplot as plt 
    # plt.figure(figsize=(30, 10))
    # plt.plot(ecg_unit[11,:])


    # import json
    # path = "/home/kruzhilov/petct/ecglib/json_doc.json"
    # f = open(path, "r")
    # data = json.loads(f.read())
    # e_number = 0
    # h_number = 0
    # for element in data["root"]:
    #     if element["id"] == "user09":
    #         if "codes" in element.keys():
    #             if "E" in element["codes"]:
    #                 e_number = e_number + 1
    #             if "H" in element["codes"]:
    #                 h_number = h_number + 1
    # print(e_number, h_number)

# %%
