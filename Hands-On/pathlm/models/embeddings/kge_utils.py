import logging
import logging.handlers
import os
import pickle
import shutil
import sys

import torch

from pathlm.knowledge_graphs.kg_utils import KG_RELATION, MAIN_PRODUCT_INTERACTION

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'
LOG_DIR = f'{ROOT_DIR}/logs'
TRANSE = 'transe'
IMPLEMENTED_KGE = [TRANSE]

def get_log_dir(dataset_name: str, embedding_name: str) -> str:
    ans = os.path.join(LOG_DIR, dataset_name, 'embeddings', embedding_name)
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_dataset_info_dir(dataset_name: str) -> str:
    ans = os.path.join(ROOT_DIR, 'data', dataset_name, 'preprocessed/mapping')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_embedding_rootdir(dataset_name: str) -> str:
    ans = os.path.join(ROOT_DIR, 'weights', dataset_name, 'embeddings')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_embedding_ckpt_rootdir(dataset_name: str) -> str:
    ans =  os.path.join(ROOT_DIR, 'weights', dataset_name, 'embeddings/ckpt')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans
def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def save_embed(dataset_name: str, embed_name: str, state_dict: dict):
    EMBEDDING_DIR = get_embedding_rootdir(dataset_name)
    embed_file = os.path.join(EMBEDDING_DIR, f'{embed_name}_embed.pkl')
    pickle.dump(state_dict, open(embed_file, 'wb'))


def load_embed(dataset_name: str, embed_name: str=None):
    EMBEDDING_DIR = get_embedding_rootdir(dataset_name)
    embed_file = os.path.join(EMBEDDING_DIR, f'{embed_name}_embed.pkl')
    print(f'Load {embed_name} embedding:', embed_file)
    if not os.path.exists(embed_file):
        # Except for file not found, raise error
        raise FileNotFoundError(f'Embedding file {embed_file} not found.')
    embed = pickle.load(open(embed_file, 'rb'))
    return embed

def load_embed_sd(dataset_name: str, embed_model_name: str=None, epoch: str=1):
    EMBEDDING_DIR = get_embedding_rootdir(dataset_name)
    embed_file = os.path.join(EMBEDDING_DIR, f'ckpt/{embed_model_name}_model_sd_epoch_{epoch}.ckpt')
    print('Load embedding:', EMBEDDING_DIR)
    if not os.path.exists(EMBEDDING_DIR):
        # Except for file not found, raise error
        raise FileNotFoundError(f'Embedding file {embed_file} not found.')
    state_dict = torch.load(embed_file, map_location=lambda storage, loc: storage)
    return state_dict

def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans

