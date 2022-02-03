import os
import numpy as np
#import torch

from opendatasets import load_wsdb, load_ann #ecg_dataset.
#from torch.utils.data import Dataset
from processing import preprocess #ecg_dataset.

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
            #annotation = load_ann(file_path[:-3] + 'hea')
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


def description_dataset(folder_pathes, save_path, code, terminate=False):
    ecg_list = []
    for folder_path in folder_pathes:
        for file_name in os.listdir(folder_path):
            if file_name[-3:] == "mat":
                file_path = os.path.join(folder_path, file_name)
                annotation = load_ann(file_path[:-3] + 'hea')
                codes = []
                for key in annotation.keys():
                    if "Dx_" in key:
                        codes.append(key)
                #ecg, _, sr = load_wsdb(file_path)
                if code in codes:
                    try:
                        ecg, _, sr = load_wsdb(file_path)
                    except:
                        print("error while reading", file_path)
                        continue
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
    #WFDB_Ga
    folder_path = "/ayb/vol1/kruzhilov/datasets/ecg/WFDB_ChapmanShaoxing"
    save_path = "/ayb/vol1/kruzhilov/datasets/ecg/Dx_164890007.npy"
    ecg_unit = description_dataset([folder_path], save_path, code="Dx_164890007", terminate=False)
    
   
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

