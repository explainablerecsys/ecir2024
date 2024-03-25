import os
from typing import Set, List, Dict

import numpy as np
import pandas as pd

from pathlm.utils import get_dataset_id2eid

SERENDIPITY = "serendipity"
COVERAGE = "coverage"
DIVERSITY = "diversity"
NOVELTY = "novelty"
CFAIRNESS = "cfairness"
PFAIRNESS = "pfairness"


def coverage(recommended_items: Set[int], n_items_in_catalog: int) -> float:
    """
    Paper: https://dl.acm.org/doi/pdf/10.1145/2926720
    Catalog coverage: Measures the rate of items in the catalog that are recommended to users in the test set.
    """
    return len(recommended_items) / n_items_in_catalog

def serendipity_at_k(user_topk: List[int], most_pop_topk: List[int], k: int) -> float:
    """
    Paper: https://dl.acm.org/doi/pdf/10.1145/2926720
    Serendipity: Proportion of items which may be surprising for the user, calculated as the the proportion of items
    recommended by  the benchmarked models that are not recommended by a prevedible baseline.
    In our case the baseline was MostPop.
    """
    user_topk, most_pop_topk = set(user_topk), set(most_pop_topk)
    intersection = user_topk.intersection(most_pop_topk)
    return (k - len(intersection)) / k


def diversity_at_k(topk_items: List[int], pid2genre: Dict[int, str]):
    """
    Paper:
    Diversity: Proportion of genres covered by the recommended items among the recommended items.
    """
    diversity_items_tok = set([pid2genre[pid] for pid in topk_items])  # set of genres
    return len(diversity_items_tok) / len(topk_items)

def novelty_at_k(topk_items: List[int], pid2popularity: Dict[int, float]):
    """
    Paper:
    Novelty: Inverse of popularity of the items recommended to the user
    """
    novelty_items_topk = [1 - pid2popularity[pid] for pid in topk_items]
    return np.mean(novelty_items_topk)

def get_item_genre(dataset_name: str) -> Dict[int, str]:
    """
    Returns a dictionary of item_id -> genre
    Note that the ids are the entity ids to be in the same space of the models.
    """
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    dataset_id2model_kg_id: Dict[str, str] = get_dataset_id2eid(dataset_name, "product")
    item_genre_df = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    item_genre_df['pid'] = item_genre_df['pid'].astype(str).map(dataset_id2model_kg_id) # Ensure 'pid' column is of type int before mapping
    item_genre_df.dropna(subset=['pid'], inplace=True) # Drop rows where 'pid' is NaN, since the KG dataset a slightly less product
    return {int(pid): genre for pid, genre in zip(item_genre_df['pid'], item_genre_df['genre'])}

def get_item_count(dataset_name: str) -> int:
    """
    Returns the number of items in the dataset
    """
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    df_items = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    return df_items.pid.unique().shape[0]

def get_item_pop(dataset_name: str) -> Dict[int, float]:
    """
    Returns a dictionary of item_id -> popularity score (0-1 normalised)
    Note that the ids are the entity ids to be in the same space of the models.
    """
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    dataset_id2model_kg_id: Dict[str, str] = get_dataset_id2eid(dataset_name, "product")
    df_items = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    df_items['pid'] = df_items['pid'].astype(str).map(dataset_id2model_kg_id) # Ensure 'pid' column is of type str before mapping
    df_items.dropna(subset=['pid'], inplace=True) # Drop rows where 'pid' is NaN, since the KG dataset a slightly less product
    return {int(pid): float(pop_item) for pid, pop_item in zip(df_items['pid'], df_items['pop_item'])}
