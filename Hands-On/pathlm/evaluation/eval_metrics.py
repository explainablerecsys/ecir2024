from typing import Dict, Tuple

from tqdm import tqdm

from pathlm.evaluation.beyond_accuracy_metrics import COVERAGE, PFAIRNESS, coverage, \
    serendipity_at_k, diversity_at_k, novelty_at_k, get_item_genre, get_item_count, get_item_pop, SERENDIPITY, \
    DIVERSITY, NOVELTY
from pathlm.evaluation.eval_utils import compute_mostpop_topk, get_precomputed_topks, REC_QUALITY_METRICS_TOPK
from pathlm.evaluation.utility_metrics import *


def print_rec_quality_metrics(avg_rec_quality_metrics: Dict[str, float], method='inline'):
    """
    args:
        avg_rec_quality_metrics: a dictionary containing the average value of each metric
    """
    if method=='latex':
        print(' & '.join(list(avg_rec_quality_metrics.keys())))
        print(' & '.join([str(round(value, 2)) for value in avg_rec_quality_metrics.values()]))
    elif method=='inline':
        print(', '.join([f'{metric}: {round(value, 2)}' for metric, value in avg_rec_quality_metrics.items()]))
    elif method=='endline':
        for metric, value in avg_rec_quality_metrics.items():
            print(f'{metric}: {round(value, 2)}')


def evaluate_rec_quality_from_results(dataset_name: str, model_name: str, test_labels: Dict[int, List[int]],
                                      k: int = 10, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    This function computes all the recommendation quality metrics for a given set of topk items that are already computed
    and stored in the results folder.
    """
    topks = get_precomputed_topks(dataset_name, model_name)
    # TOPK size is fixed to 10
    return evaluate_rec_quality(dataset_name, topks, test_labels, k, metrics)


def evaluate_rec_quality(dataset_name: str, topk_items: Dict[int, List[int]], test_labels: Dict[int, List[int]],
                         k: int = 10, method_name=None, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    This function computes all the recommendation quality metrics for a given set of topk items, please note that the topk items and test set are
    expressed using the original ids of the dataset (e.g. the ids of the movies in the MovieLens dataset).
    """
    rec_quality_metrics = {metric: list() for metric in metrics}
    recommended_items_all_user_set = set()

    n_items_in_catalog = get_item_count(dataset_name)  # Needed for coverage
    pid2popularity = get_item_pop(dataset_name)  # Needed for novelty
    pid2genre = get_item_genre(dataset_name)  # Needed for diversity
    mostpop_topk = compute_mostpop_topk(dataset_name, k)  # Needed for serendipity

    topk_sizes = []
    # Evaluate recommendation quality for users' topk
    with tqdm(desc=f"Evaluating rec quality for {method_name}", total=len(topk_items.keys())) as pbar:
        for uid, topk in topk_items.items():
            hits = []
            for pid in topk[:k]:
                hits.append(1 if pid in test_labels[uid] else 0)

            # If the model has predicted less than 10 items pad with zeros
            while len(hits) < k:
                hits.append(0)
            for metric in REC_QUALITY_METRICS_TOPK:
                if len(topk) == 0:
                    metric_value = 0.0
                else:
                    if metric == NDCG:
                        metric_value = ndcg_at_k(hits, k)
                    if metric == MRR:
                        metric_value = mmr_at_k(hits, k)
                    if metric == PRECISION:
                        metric_value = precision_at_k(hits, k)
                    if metric == RECALL:
                        test_set_len = max(max(1, len(topk)), len(test_labels[uid]))
                        metric_value = recall_at_k(hits, k, test_set_len)
                    if metric == SERENDIPITY:
                        metric_value = serendipity_at_k(topk, mostpop_topk[uid], k)
                    if metric == DIVERSITY:
                        metric_value = diversity_at_k(topk, pid2genre)
                    if metric == NOVELTY:
                        metric_value = novelty_at_k(topk, pid2popularity)
                    if metric == PFAIRNESS:
                        continue  # Skip for now
                rec_quality_metrics[metric].append(metric_value)

            # For coverage
            recommended_items_all_user_set.update(set(topk))
            topk_sizes.append(len(topk))
            pbar.update(1)

    # Compute average values for evaluation
    avg_rec_quality_metrics = {metric: np.mean(values) for metric, values in rec_quality_metrics.items()}
    avg_rec_quality_metrics[COVERAGE] = coverage(recommended_items_all_user_set, n_items_in_catalog)

    # Print results
    print(f'Number of users: {len(test_labels.keys())}, average topk size: {np.array(topk_sizes).mean():.2f}')
    print_rec_quality_metrics(avg_rec_quality_metrics)
    # print(generate_latex_row(args.model, avg_rec_quality_metrics, "rec"))
    # Save as csv if specified
    return rec_quality_metrics, avg_rec_quality_metrics


def is_faithful(path, tokenizer, kg, relid_to_type, eid_to_name):
    tokenized_path = tokenizer.convert_tokens_to_ids(path[1:])
    # Extract each time two token at the time
    for i in range(0, len(tokenized_path) - 1, 2):
        ent_u, rel, ent_v = tokenized_path[i], tokenized_path[i + 1], tokenized_path[i + 2]
        if ent_u not in kg or rel not in kg[ent_u] or ent_v not in kg[ent_u][rel]:
            if ent_u not in kg:
                fake_ent = ent_u
            elif rel not in kg[ent_u]:
                return relid_to_type[int(tokenizer.convert_ids_to_tokens(rel)[1:])]
            elif ent_v not in kg[ent_u][rel]:
                fake_ent = ent_v
            return False, eid_to_name[tokenizer.convert_ids_to_tokens(fake_ent)[1:]]
    return True, -1


"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large', help='which model to evaluate')
    parser.add_argument('--dataset', type=str, default='lfm1m', help='which dataset evaluate')
    parser.add_argument('--sample_size', default='1000', type=str,
                        help='')
    parser.add_argument('--n_hop', default='3', type=str, help='')
    args = parser.parse_args()
    
    results_dir_base = get_result_dir(args.dataset)
    custom_model_name = f'clm-end-to-end-{args.dataset}-{args.model}-{args.sample_size}-hop-{args.n_hop}'
    result_dir = os.path.join(results_dir_base, custom_model_name)
    
    test_set = get_set(args.dataset, set_str='test')
    
    with open(os.path.join(result_dir, 'topk_items.pkl'), 'rb') as f:
        topk_items = pickle.load(f)

    print("Evaluating recommendation quality our model")
    evaluate_rec_quality(topk_items, test_set)

    with open(os.path.join(result_dir, 'pred_paths.pkl'), 'rb') as f:
        our_pred_paths = pickle.load(f)

    PLM_FOLDER = os.path.join(results_dir_base, 'plm')
    with open(os.path.join(PLM_FOLDER, 'topk_items.pkl'), 'rb') as f:
        topks_plm = pickle.load(f)

    with open(os.path.join(PLM_FOLDER, 'pred_paths.pkl'), 'rb') as f:
        plm_pred_paths = pickle.load(f)

    print("Evaluating recommendation quality plm")
    evaluate_rec_quality(topks_plm, test_set)

    TOKENIZER_TYPE = "WordLevel"
    CONTEXT_LENGTH = 24
    tokenizer_dir = f'./tokenizers/{args.dataset}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=CONTEXT_LENGTH,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    print("Loading KG")
    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg_stat = KGsampler(args, args.dataset, dirpath, data_dir=data_dir_mapping)
    kg, _ = tokenize_augmented_kg(kg_stat, tokenizer)

    print("Check faithfulness")
    valid_sequences = defaultdict(list)
    for uid, paths in plm_pred_paths.items():
        for path in paths:
            tokenized_paths = tokenizer.convert_tokens_to_ids(path[1:])
            #tokenized_paths = [tokenizer.encode(path, add_special_tokens=False) for path in paths]
            #Extract each time two token at the time
            for i in range(0, len(tokenized_paths)-1, 2):
                ent_u, rel, ent_v = tokenized_paths[i], tokenized_paths[i+1], tokenized_paths[i+2]
                if ent_u not in kg or rel not in kg[ent_u] or ent_v not in kg[ent_u][rel]:
                    valid_sequences[uid].append(0.)
                    break
            else:
                valid_sequences[uid].append(1.)
    avg_rate_valid_sequences = np.mean([np.mean(valid_sequences[uid]) for uid in valid_sequences])
    print(f'Average rate of valid sequences: {avg_rate_valid_sequences}')

    eid_to_name = get_eid_to_name(args.dataset)

    #Sample 10 random users and show their predicted paths
    uids = random.sample(list(our_pred_paths.keys()), 10)
    print("========================================")
    for uid in ['372']:
        print(f'-----User {uid} predicted paths PLM:-----')
        for path in plm_pred_paths[uid]:
            decoded_path = []
            for i, token in enumerate(path):
                if i == 0: continue
                if i % 2 == 1:
                    if token.startswith('U'):
                        decoded_path.append(token)
                        continue
                    eid = token[1:]
                    decoded_path.append(eid_to_name[eid])
                else:
                    if token == 'R-1':
                        decoded_path.append('watched' if args.dataset == 'ml1m' else 'listened to')
                        continue
                    rid = int(token[1:])
                    decoded_path.append(f"--{kg_stat.rel_id2type[rid]}-->")
            decoded_path_str = ' '.join(decoded_path) + f' isFaithful {is_faithful(path, tokenizer, kg,  kg_stat.rel_id2type, eid_to_name)}'
            print(decoded_path_str)
        print(f'-----User {uid} predicted paths PERLM:-----')
        for path in our_pred_paths[uid]:
            decoded_path = []
            for i, token in enumerate(path):
                if i == 0: continue
                if i % 2 == 1:
                    if token.startswith('U'):
                        decoded_path.append(token)
                        continue
                    eid = token[1:]
                    decoded_path.append(eid_to_name[eid])
                else:
                    if token == 'R-1':
                        decoded_path.append('watched' if args.dataset == 'ml1m' else 'listened to')
                        continue
                    rid = int(token[1:])
                    decoded_path.append(f"--{kg_stat.rel_id2type[rid]}-->")
            decoded_path_str = ' '.join(decoded_path) + f' isFaithful {is_faithful(path, tokenizer, kg, kg_stat.rel_id2type, eid_to_name)}'
            print(decoded_path_str)
        print('\n')
"""
