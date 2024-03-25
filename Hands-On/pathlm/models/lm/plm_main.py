import argparse
import os
import pickle
from datasets import load_from_disk, DatasetDict, Dataset
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,\
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, LogitsProcessorList,\
    set_seed, GPT2LMHeadModel, GPT2Model, EarlyStoppingCallback


from pathlm.models.lm.plmrec import PLMRec
from pathlm.models.lm.perlm import PERLM 
from pathlm.models.lm.lm_utils import _initialise_type_masks, tokenize_augmented_kg
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.sampling import KGsampler
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.utils import SEED, check_dir, get_weight_dir

from pathlm.models.lm.trainer import PathCLMTrainer

from datetime import datetime
import wandb

# Read an example and return the tokenized version
def tokenize_function(examples: str, context_length: int = 200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)


def update_config(config, tokenizer, args):
    # Assuming _initialise_type_masks is modified to work directly with tokenizer or adjust accordingly
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)

    config.update({
        'num_hops': args.n_hop,
        'sample_size_pretrain': args.sample_size,
        'sample_size_finetune': args.sample_size,
        'sample_size_hop': args.n_hop,
        'task': args.task,
        'train_batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'ent_mask': ent_mask,
        'rel_mask': rel_mask,
        'token_id_to_token': token_id_to_token,
        # Any other configurations derived directly from args
    })
    return config


def load_embeddings(embedding_path):
    try:
        with open(embedding_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f'Embedding file not found at {embedding_path}')
        exit(-1)

def initialize_model_and_update_config(tokenizer, args):
    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': args.context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    if args.pretrain_ckpt:
        print('Loading from checkpoint for resuming training:', args.pretrain_ckpt)
        model = PLMRec.from_pretrained(args.pretrain_ckpt,
                                       config=AutoConfig.from_pretrained(args.pretrain_ckpt, **config_kwargs))
    else:
        if 'plm-rec' in args.model:
            embeds = load_embeddings(os.path.join(args.embedding_root_dir, args.dataset, 'embeddings', args.emb_filename))
            print('Using embeddings:', args.emb_filename)
            config_kwargs.update({
                'hidden_size': int(args.emb_size),
                'num_attention_heads': int(args.emb_size) // 10
            })
            config = AutoConfig.from_pretrained(args.model.split('@')[1], **config_kwargs)
            config = update_config(config, tokenizer, args)
            model = PLMRec(config)
            mapper = EmbeddingMapper(tokenizer, kg, embeds)
            mapper.init_with_embedding(model.transformer.wte.weight)
            print(f'Model {model_name} initialized with {args.emb_filename} embeddings of size: ',
                  model.transformer.wte.weight.shape)
        else:
            print("This train script must be use to train exclusevely plm-rec@<model-base> models")
            exit(-1)
    return model

def prepare_training_arguments(args):
    trainer_logging_root = os.path.join(args.output_dir, args.exp_name, 'train_checkpoints')
    check_dir(trainer_logging_root)
    return TrainingArguments(
        output_dir=trainer_logging_root,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.validation_interval,
        logging_steps=min(args.logging_interval, args.validation_interval),
        learning_rate=2e-4,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,
        save_steps=args.validation_interval,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,  # Assuming SEED value
        report_to='wandb' if args.wandb else 'none',
    )

def train(args: argparse.Namespace, tokenizer, tokenized_dataset, kg):
    model = initialize_model_and_update_config(tokenizer, args)
    print('Model config:', model.config)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)
    training_args = prepare_training_arguments(args)

    trainer = PathCLMTrainer(
        cmd_args=args,
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=args.experiment_model_name,
        logit_processor_type=args.logit_processor_type,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    weight_path = get_weight_dir(args.experiment_model_name, args.dataset)
    trainer.save_model(weight_path)

    return model

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument('--wandb', default=False, action='store_true')    
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, end-to-end}")

    parser.add_argument("--sample_size", type=str, default="250",
                        help="Number of sampled path in the chosen dataset")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")

    parser.add_argument("--logit_processor_type", type=str, default="",
                        help="Path sequence deconding method: default to Graph Constrained Decoding")
    # Model arguments
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=8, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=256, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='1', help="")
    parser.add_argument("--infer_batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=30,
                        help="Number of sequences generated for each user")
    parser.add_argument("--n_beams", type=int, default=30,
                        help="Number of sequences generated for each user")    

    # Parameter relative to resume training
    parser.add_argument("--continue_training", type=bool, default=False,
                        help="Whether to continue training from a checkpoint or not")
    parser.add_argument("--pretrain_ckpt", type=none_or_int, default=None,
                        help="Checkpoint from which to resume training of the model (default to starting from scratch)")

    # Parameter relative to weight initialization
    parser.add_argument("--embedding_root_dir", type=str, default="./weights",
                        help="default: ./weights/embeddings")
    parser.add_argument("--emb_filename", type=str, default='transe_embed.pkl', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100,
                        help="Transformer Embedding size (must match external embedding size, if chosen)")
    parser.add_argument("--logging_interval", type=int, default=100,
                        help="Logging interval of the losses")    
    parser.add_argument("--validation_interval", type=int, default=15000,
                        help="Validation interval")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of epochs")     
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="default: ./data")



    args = parser.parse_args()
    args.model = 'plm-rec@' + args.model
    set_seed(SEED)

    project_name = f'from_scratch_llm_v7@{args.dataset}'
    run_name=f"{args.exp_name}@{args.dataset}@{args.model}@{args.n_hop}@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join(project_name, run_name)
    os.makedirs(log_dir, exist_ok=True)


    dataset_dir = os.path.join(args.data_dir, args.dataset)
    args.tokenizer_dir = './tokenizers'
    args.output_dir = log_dir 
    args.experiment_model_name = f"{args.task}@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@{args.logit_processor_type}"
    
    if args.wandb:

        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=run_name,
            # track hyperparameters and run metadata
            config=vars(args)
        )
    print(args)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = os.path.join(args.tokenizer_dir, dataset_name)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    dirpath = f'{args.data_dir}/{args.dataset}/preprocessed'
    data_dir_mapping = os.path.join(args.data_dir, f'{args.dataset}/preprocessed/mapping/')
    kg = KGsampler(args.dataset, data_dir=data_dir_mapping)
    sample_size = args.sample_size
    dataset_hop_size = args.n_hop
    TOKENIZED_DATASET_PATH = os.path.join(args.data_dir, f"{dataset_name}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
    TOKEN_INDEX_PATH = os.path.join(dirpath, KGsampler.TOKEN_INDEX_FILE)
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if os.path.exists(TOKENIZED_DATASET_PATH) and os.path.exists(tokenizer_file):
        task = args.task
        tokenized_dataset = load_from_disk(
            TOKENIZED_DATASET_PATH)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
    else:
        print("Tokenizer not found, run tokenization process before training")

    # Train the model
    if args.load_model:
        # Training arguments
        curr_sample_size = args.sample_size
        custom_name = f'clm-{args.task}-{args.dataset}-{args.model}-{curr_sample_size}-{args.n_hop}-{args.logit_processor_type}/checkpoint-{args.eval_ckpt_iter}'  # f"clm-from_scratch-{args.dataset}-{args.model}"
        model = AutoModelForCausalLM.from_pretrained(
            custom_name)  
    else:
        model = train(args, tokenizer, tokenized_dataset, kg)