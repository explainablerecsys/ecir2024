import argparse
from pathlm.models.embeddings.kge_utils import get_dataset_info_dir
from transformers import set_seed
from pathlm.utils import SEED, get_data_dir
from pathlm.knowledge_graphs.kg_macros import *
from pathlm.sampling import KGsampler

def none_or_str(value):
    if value == 'None':
        return None
    return value
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    parser.add_argument('--root_dir', type=str, default='./', help='Working directory to use to compute the datasets')
    parser.add_argument('--data_dirname', type=str, default='data', help='Directory name to use to store the datasets')
    parser.add_argument('--max_n_paths', type=int, default=100, help='Max number of paths sampled for each user.')
    parser.add_argument('--max_hop', type=none_or_int, default=3, help='Max number of hops.')
    parser.add_argument("--itemset_type", type=str, default='inner', help="Choose whether final entity of a path is a product\nin the train interaction set of a user, outer set, or any reachable item {inner,outer,all} respectively")
    parser.add_argument("--collaborative", type=bool, default=False, help="Wether paths should be sampled considering users as intermediate entities")
    parser.add_argument("--with_type", type=bool, default=False, help="Typified paths")
    parser.add_argument('--nproc', type=int, default=4, help='Number of processes to sample in parallel')
    parser.add_argument("--start_type", type=none_or_str, default=USER, help="Start paths with chosen type")
    parser.add_argument("--end_type", type=none_or_str, default=PRODUCT, help="End paths with chosen type")
    args = parser.parse_args()

    set_seed(SEED)

    # root dir is current directory (according to the location from where this script is run)
    # e.g. if pathlm/sampling/main.py then ./ translates to pathlm
    ROOT_DIR = args.root_dir
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, args.data_dirname)
    SAVE_DIR = os.path.join(ROOT_DATA_DIR, 'sampled')
    # Dataset directories.

    dataset_name = args.dataset
    dirpath = get_data_dir(dataset_name)
    data_dir_mapping = get_dataset_info_dir(dataset_name)
    kg = KGsampler(args.dataset, save_dir=SAVE_DIR, data_dir=data_dir_mapping)

    MAX_HOP = args.max_hop
    N_PATHS = args.max_n_paths
    itemset_type= args.itemset_type
    COLLABORATIVE=args.collaborative
    NPROC = args.nproc
    WITH_TYPE = args.with_type
    print('Closed destination item set: ',itemset_type)
    print('Collaborative filtering: ',args.collaborative)

    LOGDIR = f'dataset_{args.dataset}__hops_{MAX_HOP}__npaths_{N_PATHS}'
    
    kg.random_walk_sampler(max_hop=MAX_HOP, logdir=LOGDIR,ignore_rels=set( ), max_paths=N_PATHS, itemset_type=itemset_type, 
        collaborative=COLLABORATIVE,
        nproc=NPROC,
        with_type=WITH_TYPE,
        start_ent_type=args.start_type,
        end_ent_type=args.end_type)
