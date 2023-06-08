import yaml
from pathlib import Path
from types import SimpleNamespace
import copy
import argparse
import os
import os
import random
import numpy as np
import torch
import time
import math


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)
    data = dictionary_to_namespace(data)
    return data


def save_config(config, path):
    config_out = copy.deepcopy(config)
    config_out = namespace_to_dictionary(config_out)
    with open(path, 'w') as file:
        yaml.dump(config_out, file, default_flow_style=False)


def load_filepaths(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)

    path_to_file = Path(filepath).parents[0]
    for key, value in data.items():
        data[key] = Path(path_to_file / Path(value)).resolve()

    data = dictionary_to_namespace(data)
    return data


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def namespace_to_dictionary(data):
    dictionary = vars(data)
    for k, v in dictionary.items():
        if type(v) is SimpleNamespace:
            v = namespace_to_dictionary(v)
        dictionary[k] = v
    return dictionary


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_run_folder(filepath):
    os.makedirs(filepath)

    logs_dir = filepath / 'logs'
    os.mkdir(logs_dir)

    checkpoints_dir = filepath / 'chkp'
    os.mkdir(checkpoints_dir)

    models_dir = filepath / 'models'
    os.mkdir(models_dir)

    return True
