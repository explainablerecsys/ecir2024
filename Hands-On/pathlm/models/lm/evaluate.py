import argparse
import os
import pickle
import random

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import LogitsProcessorList
from transformers import set_seed, PreTrainedTokenizerFast
import wandb

from pathlm.datasets.data_utils import get_eid_to_name, get_user_positives
from pathlm.evaluation.eval_metrics import evaluate_rec_quality
from pathlm.evaluation.eval_utils import get_user_negatives, get_set
from pathlm.models.lm.perlm import PERLM
from pathlm.models.lm.plmrec import PLMRec
from pathlm.models.lm.decoding_constraints import ConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel, \
    PrefixConstrainedLogitsProcessorWordLevel
from pathlm.models.lm.lm_utils import tokenize_augmented_kg, get_user_negatives_and_tokens_ids, _initialise_type_masks
from pathlm.models.lm.ranker import CumulativeSequenceScoreRanker
from pathlm.sampling import KGsampler
from pathlm.utils import get_pid_to_eid, check_dir, SEED


class Evaluator:
    def __init__(
            self,
            dataset_name: str=None,
            n_hop: int=3,
            k: int=10,
            infer_batch_size: int=1,
            n_sequences_per_user: int=10,
            tokenizer=None,
            eval_device: str='cpu',
            tokenized_kg=None,
            custom_model_name=None,
            logit_processor_type: str='gcd',
                **kwargs
    ):
        super().__init__(**kwargs)
        data_dir = f"data/{dataset_name}"
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = custom_model_name
        self.result_folder = f'results/{dataset_name}/{custom_model_name}'
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.k = k
        self.eval_device = eval_device

        self.SEQUENCE_LEN = 2 * int(n_hop) + 2  # Special tokens [BOS] included

        self.INFERENCE_BATCH_SIZE = args.infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ', self.SEQUENCE_LEN)

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_dataset_id2eid(dataset_name, 'product').values()])
        self.user_negatives, self.user_negatives_token_ids = get_user_negatives_and_tokens_ids(dataset_name, tokenizer)
        self.id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): uid for uid in uids}

        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids]}
        self.test_dataset = Dataset.from_dict(self.inference_paths)

        logit_processor = None
        logit_proc_kwargs = {}
        if logit_processor_type == 'gcd':
            logit_processor_cls = ConstrainedLogitsProcessorWordLevel 
        elif logit_processor_type == 'pgcd':
            logit_processor_cls = PrefixConstrainedLogitsProcessorWordLevel
        else:
            logit_processor_cls = PLMLogitsProcessorWordLevel 
            ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
            logit_proc_kwargs['ent_mask'] = ent_mask
            logit_proc_kwargs['rel_mask'] = rel_mask
            logit_proc_kwargs['token_id_to_token'] = token_id_to_token
        print('Using: ', logit_processor_cls)


        self.logits_processor = LogitsProcessorList([
            logit_processor_cls(tokenized_kg=tokenized_kg,
                                force_token_map=self.user_negatives_token_ids,
                                tokenizer=tokenizer,
                                total_length=self.SEQUENCE_LEN,  # LAST_TOKEN_POS,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.id_to_uid_token_map,
                                eos_token_ids=[
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)],
                                **logit_proc_kwargs
                            )
        ])

        self.ranker = CumulativeSequenceScoreRanker(tokenizer, user_negatives=self.user_negatives, K=self.k,
                                                    max_new_tokens=self.SEQUENCE_LEN-len(init_condition_fn(0).split()))
        print('Using: ', self.ranker)

    def __generate_topks_withWordLevel(self, model):
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset), batch_size):
                batch = self.test_dataset[i:i + batch_size]
                inputs = self.tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(self.eval_device)
                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN,
                    min_length=self.SEQUENCE_LEN,
                    num_return_sequences=30,
                    num_beams=30,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                self.ranker.update_topk(outputs)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in self.ranker.topk.values()) / len(self.ranker.topk))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        return self.ranker.topk, self.ranker.topk_sequences

    def _generate_topks_withWordLevel_randomUser(self, model):
        #random.seed(SEED)
        #random.sample(self.test_dataset, 1)
        user = self.test_dataset[3024]
        inputs = self.tokenizer(user["uid"], return_tensors='pt', add_special_tokens=False, ).to(
            self.eval_device)
        outputs = model.generate(
            **inputs,
            max_length=self.SEQUENCE_LEN,
            min_length=self.SEQUENCE_LEN,
            num_return_sequences=900,
            num_beams=900,
            length_penalty=0.,
            num_beam_groups=150,
            diversity_penalty=0.3,
            do_sample=False,
            # top_p=0.4,
            logits_processor=self.logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
        )
        self.ranker.custom_update_topk(outputs, True)
        for uid, paths in self.ranker.topk_sequences.items():
            last_elem_counts = {}
            for path in paths:
                m2 = path[-1]
                if m2 in last_elem_counts:
                    last_elem_counts[m2] += 1
                else:
                    last_elem_counts[m2] = 1

            # Filter paths, only keep those where the last element occurs at least 3 times
            filtered_paths = [path for path in paths if last_elem_counts[path[-1]] >= 1]

            # Update the dictionary with the filtered paths for this user
            self.ranker.topk_sequences[uid] = filtered_paths

        return self.ranker.topk, self.ranker.topk_sequences

    def load_from_pkl(self, what='topk_items'):
        return pickle.load(open(f"{self.result_folder}/{what}.pkl", "rb"))

    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok
        if os.path.exists(os.path.join(self.result_folder, f"topk_items.pkl")):
            print('Loading from previously generated topks and sequences')
            topks = self.load_from_pkl(what='topk_items')
            topk_sequences = self.load_from_pkl(what='pred_paths')
        else:
            print('Generating topks and sequences')
            topks, topk_sequences = self.__generate_topks_withWordLevel(model)
            check_dir(self.result_folder)
            pickle.dump(topks, open(f"{self.result_folder}/topk_items_{args.logit_processor_type}.pkl", "wb"))
            pickle.dump(topk_sequences, open(f"{self.result_folder}/pred_paths_{args.logit_processor_type}.pkl", "wb"))

        _, avg_rec_quality_metrics = evaluate_rec_quality(self.dataset_name, topks, self.test_set)
        return avg_rec_quality_metrics


def is_faithful(path, tokenizer, kg, relid_to_type, eid_to_name):
    tokenized_path = tokenizer.convert_tokens_to_ids(path[1:])
    # Extract each time two token at the time
    for i in range(0, len(tokenized_path) - 1, 2):
        ent_u, rel, ent_v = tokenized_path[i], tokenized_path[i + 1], tokenized_path[i + 2]
        if ent_u not in kg or rel not in kg[ent_u] or ent_v not in kg[ent_u][rel]:
            if ent_u not in kg:
                fake_ent = ent_u
            elif rel not in kg[ent_u]:
                return False, relid_to_type[int(tokenizer.convert_ids_to_tokens(rel)[1:])]
            elif ent_v not in kg[ent_u][rel]:
                fake_ent = ent_v
            return False, eid_to_name[tokenizer.convert_ids_to_tokens(fake_ent)[1:]]
    return True, -1

def paths_to_plainText(dataset_name, paths, kg_stat, tokenized_kg, tokenizer):
    eid_to_name = get_eid_to_name(dataset_name)
    faith_paths = []
    for path in paths:
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
        is_faith = is_faithful(path, tokenizer, tokenized_kg, kg_stat.rel_id2type, eid_to_name)
        decoded_path_str = ' '.join(
            decoded_path) + f' isFaithful: {is_faith[0]}, fake_ent: {is_faith[1]}'
        faith_paths.append(1 if is_faith[0] else 0)
        print(decoded_path_str)
    print(f'Number of faithful paths: {np.array(faith_paths).mean()}')

def get_best_checkpoint(model_folder):
    #get the checkpoint with the highest step number in filename
    checkpoints_filenames = [f for f in os.listdir(model_folder) if f.startswith("checkpoint")]
    checkpoints_filenames.sort()
    return checkpoints_filenames[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, finetune, end-to-end}")
    parser.add_argument("--loading_checkpoint", type=bool, default=False, help="True to load checkpoint False to load from model-weights")
    parser.add_argument("--sample_size", type=str, default="1000",
                        help="Which sample size dataset to use for fine-tuning/end-to-end")
    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2-large",
                        help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--logit_processor_type", type=str, default="None",
                        help="Path sequence deconding method: default to Graph Constrained Decoding")    
    parser.add_argument("--n_hop", type=str, default="3",
                        help="")
    parser.add_argument("--k", type=int, default=10,
                        help="Size of the top-k recommendation list")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--test_batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--infer_batch_size", type=int, default=192, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=10,
                        help="Number of sequences generated for each user at inference time")
 

    args = parser.parse_args()

    set_seed(SEED)
    
    print(f'sample_size: {args.sample_size}, model: {args.model}, n_hop: {args.n_hop}')
    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    model_custom_name = f"{args.task}@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}"
    if args.loading_checkpoint:
        model_folder = f"{args.task}@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}"
        best_checkpoint = get_best_checkpoint(model_folder)
        model_folder = f"{model_folder}/{best_checkpoint}"
        print(f'loading from ckpt: {model_folder}')
    else:
        model_folder = f"weights/{args.dataset}/{model_custom_name}"
        print("loading from model weights: ", model_folder)
    if 'plm-rec' in model_name:
        model = PLMRec.from_pretrained(model_folder).to(args.eval_device)
    else:
        model = PERLM.from_pretrained(model_folder).to(args.eval_device)


    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGsampler(args.dataset, data_dir=data_dir_mapping)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)
    
    print("Evaluating...")
    evaluator_obj = Evaluator(
        dataset_name = args.dataset,
        tokenized_kg = tokenized_kg,
        k=1000,
        n_hop = args.n_hop,
        infer_batch_size = args.infer_batch_size,
        n_sequences_per_user = args.n_seq_infer,
        tokenizer = tokenizer,
        eval_device = args.eval_device,
        custom_model_name = model_custom_name,
        logit_processor_type=args.logit_processor_type,
    )
    #evaluator_obj.evaluate(model)
    #topk, topk_sequences = evaluator_obj.ranker.topk, evaluator_obj.ranker.topk_sequences
    #paths = paths_to_plainText(args.dataset, list(topk_sequences.values())[0], kg, tokenized_kg, tokenizer)
    #print(paths)

    topk, topk_sequences = evaluator_obj._generate_topks_withWordLevel_randomUser(model)
    positives = get_user_positives(args.dataset)
    eid2name = get_eid_to_name(args.dataset)
    print(len(positives[3024]), [eid2name[str(positive)] for positive in positives[3024]])
    paths = paths_to_plainText(args.dataset, list(topk_sequences.values())[0], kg, tokenized_kg, tokenizer)
    print(paths)
    #.evaluate(model)

