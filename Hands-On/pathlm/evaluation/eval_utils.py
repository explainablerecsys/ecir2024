import os
import pickle
from collections import defaultdict, Counter
import random
from typing import Dict, List

from pathlm.utils import check_dir

from pathlm.datasets.data_utils import get_set, get_user_negatives, get_user_positives
from tqdm import tqdm
from pathlm.evaluation.utility_metrics import NDCG, MRR, PRECISION, RECALL
from pathlm.evaluation.beyond_accuracy_metrics import SERENDIPITY, DIVERSITY, NOVELTY

REC_QUALITY_METRICS_TOPK = [NDCG, MRR, PRECISION, RECALL, SERENDIPITY, DIVERSITY,
                            NOVELTY]

def save_topks_items_results(dataset_name: str, model_name: str, topk_items: Dict[int, List[int]], k: int=10):
    """
    Save the topk items for each user in the test set into a pickle file
    Note that the topk items uses the entity id, not the item id
    """
    result_dir = get_result_dir(dataset_name, model_name)
    check_dir(result_dir)
    with open(os.path.join(result_dir, f'top{k}_items.pkl'), 'wb') as f:
        pickle.dump(topk_items, f)

def save_topks_paths_results(dataset_name: str, model_name: str, topk_paths: Dict[int, List[int]], k: int=10):
    result_dir = get_result_dir(dataset_name, model_name)
    check_dir(result_dir)
    with open(os.path.join(result_dir, f'top{k}_paths.pkl'), 'wb') as f:
        pickle.dump(topk_paths, f)

def get_precomputed_topks(dataset_name: str, model_name: str, k=10) -> Dict[str, List[str]]:
    result_dir = get_result_dir(dataset_name, model_name)
    with open(os.path.join(result_dir, f'top{k}_items.pkl'), 'rb') as f:
        topk_items = pickle.load(f)
    topk_items = {int(k): [int(v) for v in topk_items[k]] for k in topk_items}
    return topk_items

def compute_random_baseline(dataset_name: str, k: int=10) -> Dict[int, List[int]]:
    test_set = get_set(dataset_name, set_str='test')
    user_negatives = get_user_negatives(dataset_name)
    topks = {}
    for uid in tqdm(list(test_set.keys()), desc="Evaluating", colour="green"):
        topks[uid] = random.sample(user_negatives[uid], k)
    return topks


def compute_mostpop_topk(dataset_name: str, k: int=10) -> Dict[int, List[int]]:
    """
    Mostpop recommender, returns the topks using the uids and pids remaped to the entity ids, so the index space is
    the same as the one used in the model. The popularity of the items is computed using the train set,
    valid items are not recommendable
    """
    train_items = get_set(dataset_name, set_str='train')
    valid_items = get_set(dataset_name, set_str='valid')
    #Computing the most popular items
    interacted_items = []
    for uid in train_items:
        interacted_items.extend(train_items[uid])
    item_frequency = sorted(Counter(interacted_items).items(), key=lambda x: x[1], reverse=True)

    #Computing the topk items
    topks = dict()
    for uid in train_items:
        topks[uid] = []
        train_item_set = set(train_items[uid])
        valid_items_set = set(valid_items[uid])
        topk_items = set()
        for pid, freq in item_frequency:
            if pid in train_item_set or pid in valid_items_set:
                continue
            if pid in topk_items:
                continue
            topks[uid].append(pid)
            topk_items.add(pid)
            if len(topks[uid]) >= k:
                break
    return topks


def get_result_base_dir(dataset_name: str) -> str:
    return os.path.join('results', dataset_name)

def get_result_dir(dataset_name: str, model_name: str) -> str:
    return os.path.join('results', dataset_name, model_name)
