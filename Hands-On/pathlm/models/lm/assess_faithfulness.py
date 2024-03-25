
import argparse
import csv
import os
import pickle
import random
from collections import defaultdict, Counter

import pandas as pd

import numpy as np
from pathlm.models.embeddings.kge_utils import get_dataset_info_dir

from pathlm.utils import get_data_dir

from pathlm.evaluation.eval_metrics import evaluate_rec_quality_from_results

from pathlm.datasets.data_utils import get_set, get_rid_to_name

from pathlm.evaluation.eval_utils import get_result_dir
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


from pathlm.models.lm.lm_utils import tokenize_augmented_kg
from pathlm.sampling import KGsampler


def print_rec_quality_metrics(avg_rec_quality_metrics):
    for metric, value in avg_rec_quality_metrics.items():
        print(f'{metric}: {round(value, 2)}')


def get_eid_to_name(dataset_name):
    eid2name = dict()
    with open(os.path.join(f'data/{dataset_name}/preprocessed/e_map.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            eid, name = row[:2]
            eid2name[eid] = ' '.join(name.split('_'))
    return eid2name

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilgpt2', help='which model to evaluate')
    parser.add_argument('--dataset', type=str, default='ml1m', help='which dataset evaluate')
    parser.add_argument('--sample_size', default='250', type=str,
                        help='')
    parser.add_argument('--n_hop', default='3', type=str, help='')
    parser.add_argument('--k', default='10', type=str, help='')
    parser.add_argument('--decoding_strategy', default='gcd', type=str, help='Empty for {PLM} or gcd for {PERLM}')
    args = parser.parse_args()

    if "plm-rec" in args.model:
        print("Inputing a PLM")
        args.model = f'end-to-end@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@'
    elif args.model in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        args.model = f'end-to-end@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@{args.decoding_strategy}'
    result_dir = get_result_dir(args.dataset, args.model)
    test_set = get_set(args.dataset, set_str='test')

    with open(os.path.join(result_dir, f'top{args.k}_items.pkl'), 'rb') as f:
        topk_items = pickle.load(f)

    #evaluate_rec_quality_from_results(args.dataset, args.model, test_set)
    

    with open(os.path.join(result_dir, f'top{args.k}_paths.pkl'), 'rb') as f:
        topks_paths = pickle.load(f)


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
    # Dataset directories.
    dirpath = get_data_dir(args.dataset)

    data_dir_mapping = get_dataset_info_dir(args.dataset)
    kg_stat = KGsampler(args.dataset, data_dir=data_dir_mapping)
    kg, _ = tokenize_augmented_kg(kg_stat, tokenizer)

    print("Check faithfulness")
    valid_sequences_counter = defaultdict(list)
    corrupted_sequences = defaultdict(list)
    broken_at_len = []
    for uid, paths in topks_paths.items():
        for path in paths:
            tokenized_paths = tokenizer.convert_tokens_to_ids(path[1:])
            #Extract each time two token at the time
            for i in range(0, len(tokenized_paths)-1, 2):
                ent_u, rel, ent_v = tokenized_paths[i], tokenized_paths[i+1], tokenized_paths[i+2]
                if ent_u not in kg or rel not in kg[ent_u] or ent_v not in kg[ent_u][rel]:
                    if ent_u not in kg:
                        fake_ent = ent_u
                        broken_at = i
                    elif rel not in kg[ent_u]:
                        fake_ent = rel # fake relation
                        broken_at = rel
                        broken_at = i + 1
                    elif ent_v not in kg[ent_u][rel]:
                        fake_ent = ent_v
                        broken_at = i + 2
                    broken_at_len.append(broken_at)
                    valid_sequences_counter[uid].append(0.)
                    corrupted_sequences[uid].append([path[1:], tokenizer.decode(fake_ent), broken_at])
                    break
            else:
                valid_sequences_counter[uid].append(1.)
    avg_rate_valid_sequences = np.mean([np.mean(valid_sequences_counter[uid]) for uid in valid_sequences_counter])
    print(f'Average rate of valid sequences per user: {avg_rate_valid_sequences}')
    print(f'{len(broken_at_len)}/{len(valid_sequences_counter) * int(args.k)} corrupted sequences, specifically at position {Counter(broken_at_len)}')
    random_uid = random.choice(list(corrupted_sequences.keys()))
    eid_to_name = get_eid_to_name(args.dataset)
    rid_to_name = get_rid_to_name(args.dataset)
    rid_to_name['-1'] = 'interacted'
    print(f'Examples of corrupted sequence for user {uid}:')
    paths = topks_paths[random_uid]

    for i, path in enumerate(paths):
        if i > len(corrupted_sequences[uid])-1: break
        corrupted_seq, corrupted_piece, _ = corrupted_sequences[uid][i]
        plain_text_path = [corrupted_seq[0]] + [eid_to_name[token[1:]] if j % 2 == 0 else rid_to_name[token[1:]] for j, token in enumerate(corrupted_seq[1:], 1)]
        plain_text_corrupted = eid_to_name[corrupted_piece[1:]] if not corrupted_piece.startswith('R') else rid_to_name[corrupted_piece[1:]]
        print("Raw corrupted sequence:")
        print(f'{corrupted_sequences[uid][i][0]}, '
              f'incoherence for {corrupted_sequences[uid][i][1]}')
        print("Plain text corrupted sequence:")
        print(f'{plain_text_path}, '
              f'incoherence for {plain_text_corrupted}')
        print("\n")
    exit(-1)
