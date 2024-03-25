from typing import List
import numpy as np

NDCG = "ndcg"
MRR = "mrr"
PRECISION = 'precision'
RECALL = 'recall'

def precision_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k] != 0
    if r.size != 0:
        return np.mean(r)
    return 0.

def recall_at_k(hit_list: List[int], k: int, test_set_len: int) -> float:
    r = np.asfarray(hit_list)[:k] != 0
    if r.size != 0:
        return np.sum(r) / test_set_len
    return 0.

def F1(pre: float, rec: float) -> float:
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def dcg_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(hit_list: List[int], k: int) -> float:
    dcg_max = dcg_at_k(sorted(hit_list, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(hit_list, k) / dcg_max

def mmr_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k]
    hit_idxs = np.nonzero(r)
    if len(hit_idxs[0]) > 0:
        return 1 / (hit_idxs[0][0] + 1)
    return 0.

