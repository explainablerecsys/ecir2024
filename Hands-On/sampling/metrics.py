import os
import argparse
from typing import Dict, List
from pathlm.models.rl.PGPR.pgpr_utils import * 
from pathlm.sampling import KGsampler



def get_set(dataset_name: str, set_str: str='test') -> Dict[str, List[int]]:
    # Get pid2eid dictionary to allow conversions from pid to eid
    def get_pid_to_eid(data_dir: str) -> dict:
        i2kg_path = os.path.join(data_dir, 'preprocessed', 'i2kg_map.txt')
        pid2eid = {}
        with open(i2kg_path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                eid = row[0]
                pid = row[1]
                pid2eid[pid] = eid
        f.close()
        return pid2eid    
    data_dir = f"../data/{dataset_name}"
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    i2kg = get_pid_to_eid(data_dir)

    # Generate paths for the test set
    curr_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/{set_str}.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = user_id  # user_id starts from 1 in the augmented graph starts from 0
            item_id = i2kg[item_id]  # Converting dataset id to eid
            curr_set[user_id].append(item_id)
    f.close()
    return curr_set


def item_coverage(stats_kg, path_dataset, end_item=True):
    n_users = len(stats_kg.user_dict)
    n_items = len(stats_kg.items)
    item_ids = set()
    for path in path_dataset:
        if end_item:
            item_ids.add(path[-1] )
        else:
            for idx in range(2,len(path)+1, 4):
                item_ids.add(path[idx])
    return len(item_ids)/n_items

def num_paths(stats_kg, path_dataset):

    return len(path_dataset)

def catalog_coverage(stats_kg, path_dataset,end_item=True):
    n_users = len(stats_kg.user_dict)
    n_items = len(stats_kg.items)

    n_inter = n_users*n_items
    #for uid in stats_kg.user_dict:
    #    inter_pids = stats_kg.user_dict[uid]
    #    n_inter += len(inter_pids)

    item_ids = set()
    for path in path_dataset:
        if end_item:
            item_ids.add(  (path[0], path[-1] )  )
        else:
            for idx in range(2,len(path)+1, 4):
                item_ids.add( (path[0], path[idx] )   )
    return len(item_ids)/n_inter
def recall_bias(stats_kg, path_dataset, test_set):
    gt_test_items = set()
    for uid in test_set:
        gt_test_items.update(test_set[uid])
    reachable_items = set()
    for path in path_dataset:
        reachable_items.add(path[-1])
        for idx in range(2,len(path)+1, 4):
            reachable_items.add(path[idx])
   
    return len(gt_test_items - reachable_items)/len(gt_test_items)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    #parser.add_argument('--dataset_path', type=str, help='One of {ml1m, lfm1m}')



    args = parser.parse_args()
    ML1M = 'ml1m'
    LFM1M ='lfm1m'
    CELL='cellphones'
    ROOT_DIR = '..'
    # Dataset directories.
    DATA_DIR = {
        ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed',
        LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed',
        CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed'
    }
    #dirpath = DATA_DIR[args.dataset]
    #randwalk_filepath = os.path.join(*dirpath.split('/')[:-1], 'paths_random_walk', 'paths.txt')

    #stats_kg = KGsampler(args.dataset, dirpath)
    kg_dict = dict()
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, 'data')

    metrics = {'item_coverage' : item_coverage, 'catalog_coverage':catalog_coverage, 'recall_bias' : recall_bias}
    with open('dataset_stats_multi_item.txt', 'w') as f_stats: 
        f_stats.write(f'Dataset_path,item_coverage,catalog_coverage,recall_bias,num_paths\n')
        for dataset_name in os.listdir(ROOT_DATA_DIR):
            if dataset_name not in DATA_DIR:
                continue
            #'''
            if dataset_name not in kg_dict:
                data_dir_mapping = os.path.join(ROOT_DIR, f'data/{dataset_name}/preprocessed/mapping/')
                args.dataset = dataset_name
                dirpath = DATA_DIR[dataset_name]
                kg = KGsampler(dataset_name, data_dir=data_dir_mapping)
                kg_dict[dataset_name] = kg

            stats_kg = kg_dict[dataset_name]
            test_set = get_set(dataset_name, set_str='test')
            dataset_path = os.path.join(ROOT_DATA_DIR, dataset_name, 'paths_random_walk')
            for filename in os.listdir(dataset_path):
                filepath = os.path.join(dataset_path, filename)
                if not filename[0].isalpha():
                    continue
                file = filename.split('.')[0]

                _,task,sample_size,hops = file.split('_')
                paths = []
                print('Loading dataset: ', filepath)
                with open(filepath) as f:
                    for line in f:
                        data = line.rstrip().split(' ')
                        paths.append(data)
                print(paths[:5])  

                res1 = item_coverage(stats_kg, paths)
                res2 = catalog_coverage(stats_kg, paths)   
                res3 = recall_bias(stats_kg, paths, test_set)  
                res4 = num_paths(stats_kg, paths)  
                f_stats.write(f'{filepath},{res1},{res2},{res3},{res4}\n')           
            #'''





    '''
    paths = []
    with open(randwalk_filepath) as f:
        for line in f:
            data = line.rstrip().split(' ')
            paths.append(data)
    print(paths[:5])


    res1 = item_coverage(stats_kg, paths)
    res2 = catalog_coverage(stats_kg, paths)
    print('Item coverage(end item only): ', res1)
    print('Catalog coverage(end item only): ', res2)
    res1 = item_coverage(stats_kg, paths, end_item=False)
    res2 = catalog_coverage(stats_kg, paths,  end_item=False)
    print('Item coverage(all items in path): ', res1)
    print('Catalog coverage(all items in path): ', res2)

    '''