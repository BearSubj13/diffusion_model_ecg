import os
import functools

import numba
import cv2
import numpy as np
import scipy
import scipy.signal

#from . import utils, open_datasets, private_datasets

def resample(ecg, shape):
    resized = cv2.resize(ecg, (shape, ecg.shape[0]))
    resized = resized.astype(ecg.dtype)
    return resized

@numba.njit
def pad_with_zeros(ecg, first=0, last=0):
    output = np.zeros((ecg.shape[0], ecg.shape[1] + first + last), dtype=ecg.dtype)
    output[:, first:-last] = ecg
    return output

@numba.njit
def crop(ecg, first=0, last=0):
    return ecg[:, first: -last]


def filter_ecg(ecg, rate, freq, mode='high', order=4):
    hb_n_freq = freq / (rate / 2)
    b, a = scipy.signal.butter(order, hb_n_freq, mode)
    filtered = scipy.signal.filtfilt(b, a, ecg)
    filtered = filtered.astype(ecg.dtype)
    return filtered    


@numba.njit
def _half_or_random(value, mode):
    if mode == 'center':
        first =  value // 2
    elif mode == 'random':
        first = np.random.randint(value)
    else:
        assert False
        
    last = value - first
    return first, last

@numba.njit
def fix_length(ecg, length, mode='center'):
    
    if ecg.shape[1] < length:
        first, last = _half_or_random(length - ecg.shape[1] , mode)  
        ecg = pad_with_zeros(ecg, first=first, last=last)
        
    elif ecg.shape[1] > length:
        first, last = _half_or_random(ecg.shape[1] - length, mode)   
        ecg = crop(ecg, first=first, last=last)
    
    return ecg


# def fix_length_center(ecg, length):
    
#     if ecg.shape[1] < length:
#         to_pad = length - ecg.shape[1]
#         before = to_pad // 2
#         after = to_pad - beforre
#         ecg = pad_with_zeros(ecg, before=before, after=after)
        
#     elif ecg.shape[1] > length:
#         to_crop = ecg.shape[1] - length
#         first = to_crop // 2
#         last = to_crop - first
#         ecg = crop(ecg, first=first, last=last)
        
#     return ecg



@numba.njit
def running_median(ecg, window):
    half = window // 2
    tmp = np.zeros((ecg.shape[0] + half * 2))
    tmp[:half] = np.mean(ecg[:window])
    tmp[-half:] = np.mean(ecg[-window:])
    tmp[half:-half] = ecg
    
    rm = np.zeros(ecg.shape, dtype=np.float32)
    for i in range(rm.shape[0]):
        rm[i] = np.mean(tmp[i:i+window])
        
    return rm

@numba.njit
def substract_running_median(ecg, window):
    rm = np.zeros(ecg.shape)
    for i in range(ecg.shape[0]):
        rm[i] = running_median(ecg[i], window) 
    return ecg - rm




def cached_load(ecg_file, params):
    
    cache_is_present = params.get('CACHE_FOLDER') is not None
    
    if cache_is_present:
    
        cache_folder = params['CACHE_FOLDER']
        preprocessin_params = ['LOW_PASS', 'HIGH_PASS', 'SR', 'SUBTRACT_RUNNING_MEAN']
#         params = {key: val for key, val in params.items() if key in preprocessin_params}
        params = {key: params[key] for key in preprocessin_params}
        hashname = utils.generate_dict_hash(params)

        cache_file_name = f'{os.path.split(ecg_file)[1]}.{hashname}.npy'
        cache_full_path = os.path.join(cache_folder, cache_file_name)

        if os.path.isfile(cache_full_path):
#             return np.load(cache_full_path)
            return _lru_cached_load(cache_full_path)
    
    if ecg_file.endswith('.edf'):
        ecg, leads, sr = private_datasets.load_edf(ecg_file)
    else:
        ecg, leads, sr = open_datasets.load_wsdb(ecg_file)
        
    ecg = preprocess(ecg, sr, params)
    
    if cache_is_present:
        np.save(cache_full_path, ecg)
        
    return ecg


@functools.lru_cache(maxsize=100000)
def _lru_cached_load(file):
    return np.load(file)



def preprocess(ecg, sr, params):
    if params['LOW_PASS'] is not None:
        ecg = filter_ecg(ecg, sr, params['LOW_PASS'], mode='low')

    if params['HIGH_PASS'] is not None:
        ecg = filter_ecg(ecg, sr, params['HIGH_PASS'], mode='high')

    if params['SR'] != sr:
        new_shape = int(ecg.shape[1] * params['SR'] / sr)
        ecg = resample(ecg, new_shape)

    if params['SUBTRACT_RUNNING_MEAN'] is not None:
        ecg = substract_running_median(ecg, params['SUBTRACT_RUNNING_MEAN'])
        
    return ecg


# @numba.njit
def fix_leads(ecg, file_leads, param_leads):
    file_leads = tuple(file_leads)
    param_leads = tuple(param_leads)
    if file_leads != param_leads:
        _ecg = list()
        for lead in param_leads:
            ix = file_leads.index(lead)
            _ecg.append(ecg[ix])
        ecg = np.stack(_ecg)
    return ecg