import argparse
from math import log

import torch
from pathlm.utils import get_weight_ckpt_dir, get_weight_dir
from tqdm import tqdm
from functools import reduce
import warnings

from pathlm.evaluation.eval_utils import save_topks_items_results, save_topks_paths_results, compute_mostpop_topk, \
    compute_random_baseline
from pathlm.knowledge_graphs.kg_macros import SELF_LOOP, USER
from pathlm.knowledge_graphs.kg_utils import KG_RELATION, MAIN_PRODUCT_INTERACTION
from pathlm.evaluation.eval_metrics import evaluate_rec_quality
from pathlm.models.embeddings.kge_utils import load_embed

from pathlm.models.rl.PGPR.kg_env import BatchKGEnvironment
from pathlm.models.rl.PGPR.train_agent import ActorCritic
from pathlm.models.rl.PGPR.pgpr_utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)


def batch_beam_search(env, model, uids, device, intrain=None, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]

        probs_max, _ = torch.max(probs, 0)
        probs_min, _ = torch.min(probs, 0)

        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)

                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[env.dataset_name][path[-1][1]][
                        relation]  # Changing according to the dataset

                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def save_output(dataset_name, pred_paths):
    extracted_path_dir = LOG_DATASET_DIR[dataset_name]  # extracted_path_dir + "/pgpr"
    if not os.path.isdir(extracted_path_dir):
        os.makedirs(extracted_path_dir)

    print("Normalizing items scores...")
    # Get min and max score to performe normalization between 0 and 1
    score_list = []
    for uid, pid in pred_paths.items():
        for pid, path_list in pred_paths[uid].items():
            for path in path_list:
                score_list.append(float(path[0]))
    min_score = min(score_list)
    max_score = max(score_list)

    print("Saving pred_paths...")
    for uid in pred_paths.keys():
        curr_pred_paths = pred_paths[uid]
        for pid in curr_pred_paths.keys():
            curr_pred_paths_for_pid = curr_pred_paths[pid]
            for i, curr_path in enumerate(curr_pred_paths_for_pid):
                path_score = pred_paths[uid][pid][i][0]
                path_prob = pred_paths[uid][pid][i][1]
                path = pred_paths[uid][pid][i][2]
                new_path_score = (float(path_score) - min_score) / (max_score - min_score)
                pred_paths[uid][pid][i] = (new_path_score, path_prob, path)
    with open(extracted_path_dir + "/pred_paths.pkl", 'wb') as pred_paths_file:
        pickle.dump(pred_paths, pred_paths_file)
    pred_paths_file.close()


def extract_paths(dataset_name, save_paths, path_file, train_labels, valid_labels, test_labels):
    embeds = load_embed(args.dataset, TRANSE)
    user_embeds = embeds[USER]
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    product = main_entity
    watched_embeds = embeds[main_relation][0]
    movie_embeds = embeds[main_entity]
    scores = np.dot(user_embeds + watched_embeds, movie_embeds.T)
    validation_pids = get_validation_pids(dataset_name)
    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}

    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != product:
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]

        if uid in valid_labels and pid in valid_labels[uid]:
            continue
        if pid in train_labels[uid]:
            continue
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    if args.save_paths:
        save_output(dataset_name, pred_paths)
    return pred_paths, scores


def evaluate_paths(dataset_name, pred_paths, emb_scores, train_labels, valid_labels, test_labels):
    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    k = 10
    best_pred_paths = {}
    for uid in pred_paths:
        if uid in train_labels:
            train_pids = set(train_labels[uid])
        if uid in valid_labels:
            valid_pids = set(valid_labels[uid])

        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            if pid in valid_pids:
                continue
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    # save_best_pred_paths(extracted_path_dir, best_pred_paths)

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    pred_paths_top10 = {}

    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top10_pids = [p[-1][2] for _, _, p in sorted_path[:k]]  # from largest to smallest
        top10_paths = [p for _, _, p in sorted_path[:k]]  # paths for the top10

        # add up to 10 pids if not enough
        if args.add_products and len(top10_pids) < k:
            train_pids = set(train_labels[uid])
            valid_pids = set(valid_labels[uid])
            cand_pids = np.argsort(emb_scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in valid_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= k:
                    break
        # end of add
        pred_labels[uid] = top10_pids[::-1]  # change order to from smallest to largest!
        pred_paths_top10[uid] = top10_paths[::-1]

    save_topks_items_results(dataset_name, MODEL, pred_labels, k)
    save_topks_paths_results(dataset_name, MODEL, pred_paths_top10, k)
    random_topk = compute_random_baseline(dataset_name, k)
    evaluate_rec_quality(dataset_name, random_topk, test_labels, method_name='Random Baseline')
    mostpop_topk = compute_mostpop_topk(dataset_name, k)
    evaluate_rec_quality(dataset_name, mostpop_topk, test_labels, method_name='MostPop Baseline')
    evaluate_rec_quality(dataset_name, pred_labels, test_labels, method_name=MODEL)


# In formula w of pi log(2 + (number of patterns of same pattern type among uv paths / total number of paths among uv paths))
def get_path_pattern_weigth(path_pattern_name, pred_uv_paths):
    n_same_path_pattern = 0
    total_paths = len(pred_uv_paths)
    for path in pred_uv_paths:
        if path_pattern_name == get_path_pattern(path):
            n_same_path_pattern += 1
    return log(2 + (n_same_path_pattern / total_paths))


def test(args):
    policy_file = args.weight_dir_ckpt + '/policy_model_epoch_{}.ckpt'.format(args.epochs)

    path_file = args.weight_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    valid_labels = load_labels(args.dataset, 'valid')
    test_labels = load_labels(args.dataset, 'test')
    # kg = load_kg(args.dataset)
    if args.run_path:
        predict_paths(policy_file, path_file, args)
    if args.save_paths or args.run_eval:
        pred_paths, scores = extract_paths(args.dataset, args.save_paths, path_file, train_labels, valid_labels,
                                           test_labels)
    if args.run_eval:
        evaluate_paths(args.dataset, pred_paths, scores, train_labels, valid_labels, test_labels)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=list, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--save_paths', type=boolean, default=False, help='Save paths')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    args.weight_dir = get_weight_dir("pgpr", args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir("pgpr", args.dataset)
    test(args)