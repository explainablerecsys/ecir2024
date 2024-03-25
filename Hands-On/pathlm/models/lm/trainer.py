import os
import pickle
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import Trainer, LogitsProcessorList, PreTrainedTokenizerFast, is_torch_tpu_available
import wandb

from pathlm.evaluation.eval_metrics import evaluate_rec_quality
from pathlm.evaluation.eval_utils import get_user_negatives, get_set, save_topks_items_results, save_topks_paths_results
from pathlm.evaluation.utility_metrics import ndcg_at_k, mmr_at_k
from pathlm.models.lm.decoding_constraints import ConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel, \
    PrefixConstrainedLogitsProcessorWordLevel
from pathlm.models.lm.lm_utils import get_user_negatives_and_tokens_ids, \
    _initialise_type_masks, get_user_positives
from pathlm.models.lm.ranker import CumulativeSequenceScoreRanker
from pathlm.utils import get_dataset_id2eid, check_dir


class PathCLMTrainer(Trainer):
    def __init__(
            self,
            cmd_args=None,
            dataset_name=None,
            n_hop=3,
            infer_batch_size=1,
            n_sequences_per_user=10,
            n_beams=30,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            experiment_name=None,
            logit_processor_type='gcd',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cmd_args = cmd_args
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.eval_device = eval_device

        self.SEQUENCE_LEN = 2 * n_hop + 2  # Special tokens [BOS] included

        self.N_RET_SEQ = n_sequences_per_user
        self.N_BEAMS = n_beams
        self.INFERENCE_BATCH_SIZE = infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ', self.SEQUENCE_LEN)

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_dataset_id2eid(dataset_name, 'product').values()])
        self.user_negatives, self.user_negatives_token_ids = get_user_negatives_and_tokens_ids(dataset_name, tokenizer)
        self.token_id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): uid for uid in uids}
        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids]}

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
                                total_length=self.SEQUENCE_LEN,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.token_id_to_uid_token_map,
                                eos_token_ids=[
                                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)],
                                **logit_proc_kwargs
                                )
        ])
        self.ranker = CumulativeSequenceScoreRanker(tokenizer, user_negatives=self.user_negatives, K=10,
                                                    max_new_tokens=self.SEQUENCE_LEN-len(init_condition_fn(0).split()))
        self.test_dataset = Dataset.from_dict(self.inference_paths)

    def __generate_topks_withWordLevel(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset), batch_size):
                batch = self.test_dataset[i:i + batch_size]
                inputs = self.tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(
                    self.eval_device)

                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN,
                    min_length=self.SEQUENCE_LEN,
                    num_return_sequences=self.N_RET_SEQ,
                    num_beams=self.N_BEAMS,
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
        print("Average topk length:", sum(len(v) for v in self.ranker.topk.values()) / max(len(self.ranker.topk), 1)  ) 
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        topks, topk_sequences = self.ranker.topk, self.ranker.topk_sequences
        save_topks_items_results(self.dataset_name, self.experiment_name, topks, self.ranker.K)
        save_topks_paths_results(self.dataset_name,  self.experiment_name, topk_sequences, self.ranker.K)
        self.ranker.reset_topks()
           
        return topks

    def evaluate(self, model):
        # Generate paths for the test users
        # This heuristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks = self.__generate_topks_withWordLevel(model)
        metrics_ = dict()
       
        _, avg_rec_quality_metrics = evaluate_rec_quality(self.dataset_name, topks, self.test_set)
        for k in avg_rec_quality_metrics:
            metrics_[f'eval_{k}'] = np.mean(avg_rec_quality_metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        logs: Dict[str, float] = {}
        if self.control.should_log:
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            logs.update(metrics)
            if self.cmd_args.wandb:
                wandb.log(logs)            
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that evaluation are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.cmd_args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.cmd_args, self.state, self.control)

        # finish logging results
        if self.control.should_log:
            self.log(logs)


class PathMLMTrainer(Trainer):
    def __init__(
            self,
            dataset_name=None,
            tokenizer=None,
            context_length=None,
            eval_device='cpu',
            **kwargs
    ):
        super().__init__(**kwargs)

        data_dir = f"data/{dataset_name}"
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        self.uids = list(self.test_set.keys())
        self.eval_device = eval_device

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_dataset_id2eid(data_dir).values()])
        self.user_positives = get_user_positives(dataset_name)

        init_condition_fn = lambda uid: f"U{uid} -1 [MASK] [MASK] [MASK] [MASK]"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in self.uids]}

    def __generate_topks_withWordLevel(self, model):
        """
        Recommendation and explanation generation
        """
        # set_seed(SEED)
        dataset_name = self.dataset
        data_dir = f"data/{dataset_name}"
        tokenizer_dir = f'./tokenizers/{dataset_name}'
        TOKENIZER_TYPE = "WordLevel"

        tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=self.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)

        init_condition_fn = lambda uid: f"U{uid} R-1 [MASK] [MASK] [MASK] [MASK] [MASK]"
        user_positives = get_user_positives(dataset_name)
        sequences = [init_condition_fn(uid) for uid in self.uids]
        dataset = Dataset.from_dict({'uid': self.uids, 'sequence': sequences})
        
        topks = {}
        for uid, sequence in zip(self.uids, dataset['sequence']):
            # Tokenize the sequence and send the tensors to the same device as your model
            inputs = tokenizer(sequence, return_tensors="pt").to("cuda")

            with torch.no_grad():  # Deactivate gradients for the following code block
                # Get the model's predictions
                outputs = model(**inputs)
                predictions = outputs.logits

            # The position of the last [MASK] token is -2.
            PRODUCT_MASK_POSITION = -2

            # Select top-k predictions from each head for the last MASK
            top_k_entities = torch.topk(predictions[0, PRODUCT_MASK_POSITION], 200).indices

            # Convert token IDs to tokens
            top_k_entities = [tokenizer.decode([idx]) for idx in top_k_entities]
            top_k_entities = [x[1:] for x in top_k_entities if x[0] == 'P']
            topks[str(uid)] = list(set(top_k_entities) - set(user_positives[str(uid)]))[:10]

        return topks

    def evaluate(self, model):
        # Generate paths for the test users
        topks = self.__generate_topks_withWordLevel(model)
        check_dir(f"./results/{self.dataset_name}/{self.custom_model_name}")
        pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.custom_model_name}/topks.pkl", "wb"))
        metrics = {"ndcg": [], "mmr": [], }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(
            f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
        metrics_ = dict()
        for k in metrics:
            metrics_[f'eval_{k}'] = np.mean(metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that evaluation are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
