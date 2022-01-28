import hashlib
import urllib
from tqdm import tqdm
import tarfile
import pyunpack
import base64

import numpy as np
import pandas as pd
import sklearn

from iterstrat.ml_stratifiers import IterativeStratification
from sklearn.model_selection import StratifiedKFold

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def generate_string_hash(some_string, cut=6):
    result = hashlib.md5(some_string.encode()).hexdigest()
    result = str(result)[:cut]
    return result    


def generate_list_hash(some_list, cut=6):
    some_list = [str(el) for el in some_list]
    some_list = sorted(some_list)
    some_string = ''.join(some_list)
    return generate_string_hash(some_string, cut=cut)


def generate_dict_hash(some_dict, cut=6):
    some_list = list()
    keys = sorted(list(some_dict.keys()))
    for key in keys:
        some_list.append(key)
        some_list.append(some_dict[key])
    return generate_list_hash(some_list, cut=cut)


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize
        
        
        
def download_dataset(url, filename):    
    print('Downloading')
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename, t.update_to)       

def extract_dataset(filename, path):
    print('Extracting') 
    if filename.endswith('.tar') or filename.endswith('.tar.gz'): 
        with tarfile.open(name=filename) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                tar.extract(member=member, path=path)
        print('Extracting done')
            
    elif filename.endswith('.rar'):
        pyunpack.Archive(filename).extractall(path)
        print('Extracting done')
        
    else:
        assert False, 'Unknow archieve'
        
        
        
def encode_base64(arr):
    arr = arr.flatten().tobytes()
    encoded = base64.standard_b64encode(arr)
    encoded = encoded.decode('ascii')
    return encoded


def decode_base64(b64_string):
    r = base64.decodebytes(b64_string.encode())
    ecg = np.frombuffer(r, dtype=np.float32).reshape((12, -1))
    return ecg



def valid_split(y, valid_size, seed=None):
    pseudo_n_folds = int(np.around(1 / valid_size).astype('int32'))  
    result = kfold_cv(y, pseudo_n_folds, seed=seed)
    result = result > 0
    return result
    
    
def kfold_cv(y, n_folds, seed=None):
    if y.shape[1] > 1 and y.sum():
        r = np.array([1 / n_folds] * n_folds)
        random_state = sklearn.utils.check_random_state(seed)
        result = IterativeStratification(y, r, random_state)
    else:
        result = np.zeros(y.shape[0], dtype='int32')
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for fold_num, (train_index, test_index) in enumerate(kfold.split(y, y)):
            result[test_index] = fold_num
    return result 



def form_strat_matrix(df, params):
    stratification  = pd.DataFrame()

    if df['dataset'].nunique() > 1:
        stratification = pd.concat([stratification, pd.get_dummies(df['dataset'])], axis=1)

    if params['STRATIFICATION_DXS'] is not None:
        stratification = pd.concat([stratification, df[list(params['STRATIFICATION_DXS'])]], axis=1)
        
    return stratification.values