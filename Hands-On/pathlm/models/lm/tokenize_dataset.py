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

from pathlm.models.lm.path_dataset import PathDataset
from pathlm.sampling import KGsampler
from pathlm.utils import SEED, check_dir, get_data_dir, get_root_data_dir

# Read an example and return the tokenized version
def tokenize_function(examples: str, context_length: int = 200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, end-to-end}")
    parser.add_argument("--sample_size", type=str, default="250",
                        help="Number of sampled path in the chosen dataset")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--nproc", type=int, default=8, help="Number of processes for dataset mapping")

    args = parser.parse_args()

    set_seed(SEED)

    dataset_root_dir = get_root_data_dir(args.dataset)
    args.tokenizer_dir = './tokenizers'
    TOKENIZER_TYPE = "WordLevel"
    dataset_name = args.dataset

    tokenizer_dir = os.path.join(args.tokenizer_dir, dataset_name)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    dirpath = get_data_dir(dataset_name)
    data_dir_mapping = os.path.join(dirpath, f'mapping/')
    kg = KGsampler(args.dataset)
    sample_size = args.sample_size
    dataset_hop_size = args.n_hop
    TOKENIZED_DATASET_PATH = os.path.join(dataset_root_dir, f"{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
    TOKEN_INDEX_PATH = os.path.join(dirpath, KGsampler.TOKEN_INDEX_FILE)
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    plain_text_path = True

    print("Loading and processing path sequences...")
    dataset = PathDataset(dataset_name, dataset_root_dir, task=args.task, sample_size=sample_size, n_hop=dataset_hop_size,
                          plain_text_path=plain_text_path)

    dataset.show_random_examples()
    dataset = dataset.dataset
    print(type(dataset))

    # Word level tokenizer
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)

    tokens = []
    with open(TOKEN_INDEX_PATH) as f:
        for line in f:
            tokens.append(line.rstrip())
    tokenizer.train_from_iterator(tokens, #dataset["path"],
                                    trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS]:0 $A:0 [EOS]:0",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    )

    tokenizer.save(tokenizer_file)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    num_proc=args.nproc,
                                    remove_columns=["path"]
                                    )
    tokenized_dataset = DatasetDict({
        "train": tokenized_dataset,
    })
    # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
    check_dir(TOKENIZED_DATASET_PATH)
    tokenized_dataset.save_to_disk(
        TOKENIZED_DATASET_PATH)
