import os
import random
import argparse
import pickle
import numpy as np
import gzip
import torch
import sys

from pathlm.utils import get_weight_dir, get_weight_ckpt_dir

from pathlm.knowledge_graphs.kg_macros import ML1M, LFM1M, CELL, COCO

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

CAFE = 'cafe'
MODEL = CAFE
# Dataset directories.
DATA_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
    CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed/{MODEL}',
    COCO: f'{ROOT_DIR}/data/{COCO}/preprocessed/{MODEL}',
}
OPTIM_HPARAMS_METRIC = 'avg_valid_loss'
VALID_METRICS_FILE_NAME = 'valid_metrics.json'


LOG_DIR = f'{ROOT_DIR}/results'


LOG_DATASET_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/{MODEL}',
    LFM1M: f'{LOG_DIR}/{LFM1M}/{MODEL}',
    CELL: f'{LOG_DIR}/{CELL}/{MODEL}',
    COCO: f'{LOG_DIR}/{COCO}/{MODEL}',
}

# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/hparams_cfg',
    CELL: f'{LOG_DATASET_DIR[CELL]}/hparams_cfg',
    COCO: f'{LOG_DATASET_DIR[COCO]}/hparams_cfg',
}
BEST_CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/best_hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/best_hparams_cfg',
    CELL: f'{LOG_DATASET_DIR[CELL]}/best_hparams_cfg',
    COCO: f'{LOG_DATASET_DIR[COCO]}/best_hparams_cfg',
}
TEST_METRICS_FILE_NAME = 'test_metrics.json'
RECOM_METRICS_FILE_NAME = 'recommender_metrics.json'
RECOM_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{RECOM_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{RECOM_METRICS_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{RECOM_METRICS_FILE_NAME}',
    COCO: f'{CFG_DIR[COCO]}/{RECOM_METRICS_FILE_NAME}',
}

TEST_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{TEST_METRICS_FILE_NAME}',
    COCO: f'{CFG_DIR[COCO]}/{TEST_METRICS_FILE_NAME}',
}
BEST_TEST_METRICS_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{TEST_METRICS_FILE_NAME}',
    COCO: f'{BEST_CFG_DIR[COCO]}/{TEST_METRICS_FILE_NAME}',
}


CONFIG_FILE_NAME = 'config.json'
CFG_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{CONFIG_FILE_NAME}',
    COCO: f'{CFG_DIR[COCO]}/{CONFIG_FILE_NAME}',
}
BEST_CFG_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{CONFIG_FILE_NAME}',
    COCO: f'{BEST_CFG_DIR[COCO]}/{CONFIG_FILE_NAME}',
}

HPARAMS_FILE = f'{MODEL}_hparams_file.json'

# Model result directories.
TMP_DIR = {
    ML1M: f'{DATA_DIR[ML1M]}/tmp',
    LFM1M: f'{DATA_DIR[LFM1M]}/tmp',
    CELL: f'{DATA_DIR[CELL]}/tmp',
    COCO: f'{DATA_DIR[COCO]}/tmp',
}

LABEL_FILE = {
    ML1M: (DATA_DIR[ML1M] + '/train.txt.gz', DATA_DIR[ML1M] + '/valid.txt.gz', DATA_DIR[ML1M] + '/test.txt.gz'),
    LFM1M: (DATA_DIR[LFM1M] + '/train.txt.gz', DATA_DIR[LFM1M] + '/valid.txt.gz', DATA_DIR[LFM1M] + '/test.txt.gz'),
    CELL: (DATA_DIR[CELL] + '/train.txt.gz', DATA_DIR[CELL] + '/valid.txt.gz', DATA_DIR[CELL] + '/test.txt.gz'),
    COCO: (DATA_DIR[COCO] + '/train.txt.gz', DATA_DIR[COCO] + '/valid.txt.gz', DATA_DIR[COCO] + '/test.txt.gz'),
}


def parse_args():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ml1m', help='dataset name. One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='neural_symbolic_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device.')

    # Hyperparamters for training neural-symbolic model.
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=100, help='Number of steps for checkpoint.')
    parser.add_argument('--embed_size', type=int, default=100, help='KG embedding size.')
    parser.add_argument('--deep_module', type=boolean, default=True, help='Use deep module or not')
    parser.add_argument('--use_dropout', type=boolean, default=True, help='use dropout or not.')
    parser.add_argument('--rank_weight', type=float, default=10, help='weighting factor for ranking loss.')
    parser.add_argument('--topk_candidates', type=int, default=10, help='weighting factor for ranking loss.')

    # Hyperparameters for execute neural programs (inference).
    parser.add_argument('--k', type=int, default=10, help='size of recommendation list.')
    parser.add_argument('--sample_size', type=int, default=50, help='sample size for model.')
    parser.add_argument('--do_infer', type=boolean, default=True, help='whether to infer paths after training.')
    parser.add_argument('--do_execute', type=boolean, default=True, help='whether to execute neural programs.')
    parser.add_argument('--do_validation', type=bool, default=True, help='Whether to perform validation')
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument("--save_interval", default=10, type=int, help="Interval to save model weights.")
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )  

    args = parser.parse_args()

    # This is model directory.
    args.log_dir = f'{TMP_DIR[args.dataset]}/{args.name}'
    args.weight_dir = get_weight_dir(CAFE, args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir(CAFE, args.dataset)
    # This is the checkpoint name of the trained neural-symbolic model.
    args.symbolic_model = f'{args.weight_dir_ckpt}/symbolic_model_epoch{args.epochs}.ckpt'

    # This is the filename of the paths inferred by the trained neural-symbolic model.
    args.infer_path_data = f'{args.weight_dir}/infer_path_data.pkl'

    # Set GPU device.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.enabled = False
    set_random_seed(args.seed)

    return args


def save_embed(dataset, embed):
    if not os.path.isdir(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    embed_file = TMP_DIR[dataset] + '/embed.pkl'
    pickle.dump(embed, open(embed_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(embed_file)}".')


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(kg_file)}".')


def load_user_products(dataset, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'rb') as f:
        up = np.load(f)
    return up


def save_user_products(dataset, up, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'wb') as f:
        np.save(f, up)
    print(f'File is saved to "{os.path.abspath(up_file)}".')


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABEL_FILE[dataset][0]
    elif mode == 'valid':
        label_file = LABEL_FILE[dataset][1]
    elif mode == 'test':
        label_file = LABEL_FILE[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    # user_products = pickle.load(open(label_file, 'rb'))
    labels = {}  # key: user_id, value: list of item IDs.
    with gzip.open(label_file, 'rb') as f:
        for line in f:
            cells = line.decode().strip().split('\t')
            labels[int(cells[0])] = [int(x) for x in cells[1:]]
    return labels


def load_path_count(dataset):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    count = pickle.load(open(count_file, 'rb'))
    return count


def save_path_count(dataset, count):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    pickle.dump(count, open(count_file, 'wb'))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)
