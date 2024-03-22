import gzip
import json
import pandas as pd

# Datasets
ML1M = "ml1m"
LFM1M = "lfm1m"
CELL = "cell"

DATASETS = [ML1M, LFM1M, CELL]
AMAZON_DATASETS = [CELL]

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
USER = 'user'

# Type of entities interacted with user feedback
PRODUCT = 'product'
WORD = 'word'

MAIN_INTERACTION = {
    ML1M: "watched",
    LFM1M: "listened",
}
# Sensible attributes
GENDER = "gender"
AGE = "age"
OVERALL = "overall"

PGPR = 'pgpr'
CAFE = 'cafe'
PATH_REASONING_METHODS = [PGPR, CAFE]
TRANSE = 'transe'
EMBEDDING_METHODS = [TRANSE]

def ensure_dataset_name(dataset_name):
    if dataset_name not in DATASETS:
        print("Dataset not recognised, check for typos")
        exit(-1)
    return


def get_raw_data_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/"


def get_raw_kg_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/kg/"


def get_data_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/preprocessed/"


def get_tmp_dir(dataset_name, model_name):
    return os.path.join(get_data_dir(dataset_name), model_name, "tmp")


def get_result_dir(dataset_name, model_name=None):
    ensure_dataset_name(dataset_name)
    if model_name == None:
        return f"results/{dataset_name}/"
    return f"results/{dataset_name}/{model_name}/"


def get_model_data_dir(model_name, dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/preprocessed/{model_name}/"

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def get_dataframe_from_json(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')