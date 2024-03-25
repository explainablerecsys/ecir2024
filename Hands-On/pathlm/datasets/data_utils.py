import csv
import gzip
import os
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from pathlm.utils import get_dataset_id2eid


def get_user_negatives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user negatives in the dataset, this means the items not interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
    """
    pid2eid = get_dataset_id2eid(dataset_name, what='product')
    ikg_ids = set([int(eid) for eid in set(pid2eid.values())]) # All the ids of products in the kg
    uid_negatives = {}
    # Generate paths for the test set
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_negatives[uid] = [int(pid) for pid in list(set(ikg_ids - set(train_set[uid]) - set(valid_set[uid])))]
    return uid_negatives


def get_set(dataset_name: str, set_str: str = 'test') -> Dict[int, List[int]]:
    """
    Returns a dictionary containing the user interactions in the selected set {train, valid, test}.
    Note that the ids are the entity ids to be in the same space of the models.
    """
    data_dir = f"data/{dataset_name}"
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    uid2eid = get_dataset_id2eid(dataset_name, what='user')
    pid2eid = get_dataset_id2eid(dataset_name, what='product')

    # Generate paths for the test set
    curr_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/{set_str}.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = int(uid2eid[user_id])  # user_id starts from 1 in the augmented graph starts from 0
            item_id = int(pid2eid[item_id])  # Converting dataset id to eid
            curr_set[user_id].append(item_id)
    f.close()
    return curr_set


def get_user_positives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user positives in the dataset, this means the items interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
    """
    uid_positives = {}
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_positives[uid] = list(set(train_set[uid]).union(set(valid_set[uid])))
    return uid_positives

def get_eid_to_name(dataset_name: str) -> Dict[str, str]:
    eid2name = dict()
    with open(os.path.join(f'data/{dataset_name}/preprocessed/e_map.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            eid, name = row[:2]
            eid2name[eid] = ' '.join(name.split('_'))
    return eid2name

def get_local_eid_to_name(dataset: str) -> Dict[str, Dict[str, str]]:
    """
    Return a dict where the keys are the type of entity and values are dicts that map local id the name of the entity
    Requires CAFE map_dataset.py execution
    """
    entity2plain_text_map = defaultdict(dict)
    with gzip.open(f"data/{dataset}/preprocessed/cafe/kg_entities.txt.gz", 'rt') as entities_file:
        reader = csv.reader(entities_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            row[1] = row[1].split("_")
            entity_type, local_id = '_'.join(row[1][:-1]), row[1][-1]
            entity2plain_text_map[entity_type][int(local_id)] = row[-1]
    entities_file.close()
    return entity2plain_text_map


def get_rid_to_name(dataset_name: str) -> Dict[str, str]:
    rid2name = dict()
    with open(os.path.join(f'data/{dataset_name}/preprocessed/r_map.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            rid, name = row[0], row[2]
            rid2name[rid] = name
    return rid2name
