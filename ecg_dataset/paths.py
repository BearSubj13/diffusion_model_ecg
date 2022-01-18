import os.path

REPO_PATH = '/mnt/kyegorov/sbermed/ecg'
DATA_PATH = '/mnt/datasets/ecg_data'

DATASETS_DIR = os.path.join(DATA_PATH, 'datasets')
WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')
LOGS_DIR = os.path.join(REPO_PATH, 'logs')
LIB_PATH = os.path.join(REPO_PATH, 'ecglib')