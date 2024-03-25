import csv
from os.path import join
import os
from typing import Dict

SEED = 2023

import torch 
import random
import numpy as np

def check_dir(dir_path: str) -> None:
    """
    Check if directory exists and create it if not
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def normalise_name(name: str) -> str:
    """
    Clean entities name from _ or previxes
    """
    if name.startswith("Category:"):
        name = name.replace("Category:", "")
    return name.replace("_", " ")

def set_seed(seed=SEED, use_deterministic=True):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic)
    np.random.seed(seed) 
    random.seed(seed)

def get_dataset_id2eid(dataset_name: str, what: str="user") -> Dict[str, str]:
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    file = open(os.path.join(data_dir, f"mapping/{what}.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    dataset_pid2eid = {}
    next(csv_reader, None)
    for row in csv_reader:
        dataset_pid2eid[row[1]] = row[0]
    file.close()
    return dataset_pid2eid

def get_eid2dataset_id(dataset_name: str, what: str="user") -> Dict[str, str]:
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    file = open(os.path.join(data_dir, f"mapping/{what}.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    eid2dataset_id = {}
    next(csv_reader, None)
    for row in csv_reader:
        eid2dataset_id[row[0]] = row[1]
    file.close()
    return eid2dataset_id

def get_rid_to_name_map(dataset_name: str) -> dict:
    """
    Get rid2name dictionary to allow conversion from rid to name
    """
    r_map_path = join(f'data/{dataset_name}', 'preprocessed/r_map.txt')
    rid2name = {}
    with open(r_map_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            rid = row[0]
            rname = normalise_name(row[-1])
            rid2name[rid] = rname
    f.close()
    return rid2name

def get_data_dir(dataset_name: str) -> str:
    return join('data', dataset_name, 'preprocessed')

def get_root_data_dir(dataset_name: str) -> str:
    return join('data', dataset_name)

def get_model_data_dir(model_name: str, dataset_name: str) -> str:
    return join(get_data_dir(dataset_name), model_name)

def get_weight_dir(model_name: str, dataset_name: str) -> str:
    weight_dir_path = join('weights', dataset_name, model_name)
    check_dir(weight_dir_path)
    return weight_dir_path

def get_weight_ckpt_dir(model_name: str, dataset_name: str) -> str:
    weight_ckpt_dir_path = join(get_weight_dir(model_name, dataset_name), 'ckpt')
    check_dir(weight_ckpt_dir_path)
    return weight_ckpt_dir_path


def get_eid_to_name_map(dataset_name: str) -> Dict[str, str]:
    eid2name = dict()
    with open(os.path.join(f'data/{dataset_name}/preprocessed/e_map.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            eid, name = row[:2]
            eid2name[eid] = ' '.join(name.split('_'))
    return eid2name

