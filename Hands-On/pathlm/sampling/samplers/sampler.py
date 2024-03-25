import multiprocessing as mp
from collections import defaultdict   
import numpy as np
from pathlm.utils import get_data_dir
from tqdm import tqdm
import pandas as pd
import json
import os
import pickle
import itertools
import functools
import pandas as pd
from collections import deque
import random

from pathlm.datasets.data_utils import get_set
from pathlm.knowledge_graphs.kg_macros import USER, ENTITY, PRODUCT, INTERACTION
from pathlm.knowledge_graphs.kg_utils import MAIN_PRODUCT_INTERACTION, KG_RELATION, PATH_PATTERN
from pathlm.models.rl.PGPR.pgpr_utils import *
from pathlm.datasets.kg_dataset_base import KARSDataset
from .constants import LiteralPath


def random_walk_typified(uid, dataset_name, kg, items, n_hop, KG2T, R2T, USER_ENT, PROD_ENT, EXT_ENT, U2P_REL, logdir, 
         user_dict, ignore_rels=set(), max_paths=None, itemset_type='inner', REL_TYPE2ID=None, 
         collaborative=True,
         num_beams=10, scorer=None,
         dataset_info=None, 
         with_type=True,
         start_ent_type=USER,
         end_ent_type=PRODUCT):
    dirpath = logdir
    os.makedirs(dirpath, exist_ok=True)

    REL_TYPE2ID[U2P_REL] = LiteralPath.interaction_rel_id
    u2p_rel_id = LiteralPath.interaction_rel_id
    user_prod_cache = dict()
    unique_path_set = set()    
    def dfs(uid, prev_ent_t, cur_ent_t, prev_ent_id, cur_ent_id, cur_hop, path, prev_rel, n_hop, start_ent_type, end_ent_type,
                                    cur_attempts,
                                    max_attempts=8000): # orig is 2k
        if cur_hop >= n_hop:
            path = [ str(x) for x in path]
            path_str = ' '.join(path)
            if path_str in unique_path_set:
                return False
            else:
                unique_path_set.add(path_str)

            fp.write(path_str + '\n' )  
            
            return True



        if cur_ent_id not in kg[cur_ent_t]:
            return False
        valid_rels = list(kg[cur_ent_t][cur_ent_id].keys())
        random.shuffle(valid_rels)
        for rel in valid_rels:
            if rel in ignore_rels:
                continue
            rel_id = REL_TYPE2ID[rel]

            candidate_types = kg[cur_ent_t][cur_ent_id][rel]


            random.shuffle(candidate_types)
            for cand_type in candidate_types:
                if not collaborative and cand_type == USER_ENT:
                    continue 
                candidates = kg[cur_ent_t][cur_ent_id][rel][cand_type]

                if cur_hop == n_hop-1 and end_ent_type == PROD_ENT:
                    cache_key = (uid, rel, cand_type)
                    if itemset_type == 'inner':
                        if cache_key not in user_prod_cache:
                            candidates = list(user_dict[uid].intersection(set(candidates)))
                            user_prod_cache[cache_key] = candidates
                        else:
                            candidates = user_prod_cache[cache_key]
                    elif itemset_type == 'outer':
                        if cache_key not in user_prod_cache:
                            candidates = list(set(candidates) - user_dict[uid]   )
                            user_prod_cache[cache_key] = candidates
                        else:
                            candidates = user_prod_cache[cache_key]                                            
                    elif itemset_type == 'all':
                        # candidate set is left unchanged
                        pass
                    else:
                        continue

                    ent_t = None
                    if cur_ent_t == USER_ENT:
                        ent_t = USER_ENT
                    else:
                        ent_t = EXT_ENT
                    key = uid, rel_id, ent_t, cur_ent_id

                if len(candidates) == 0:
                    continue         
                if with_type:    
                    if rel_id == prev_rel:
                        path.append( LiteralPath.back_rel )
                    else:
                        path.append( LiteralPath.fw_rel )
                path.append( f'{LiteralPath.rel_type}{rel_id}' )
                random.shuffle(candidates)
                

                for next_ent_id in candidates:
                    if next_ent_id == prev_ent_id and prev_ent_t == cand_type:
                        continue
                    if cand_type == USER_ENT:
                        if next_ent_id == uid:
                            prefix = LiteralPath.main_user
                        else:
                            prefix = LiteralPath.oth_user
                        type_prefix = LiteralPath.user_type

                    elif cand_type == PROD_ENT:#next_ent_id not in items:
                        if cur_hop == n_hop-1 and end_ent_type == PROD_ENT:
                            assert next_ent_id in user_dict[uid], f'Error: {next_ent_id} not found for user: {uid} in {user_dict[uid]}' 
                        if next_ent_id in user_dict[uid]:  
                            prefix = LiteralPath.recom_prod
                        else:
                            prefix = LiteralPath.prod    
                        type_prefix = LiteralPath.prod_type                  
                    else:              
                        prefix = LiteralPath.ent
                        type_prefix = LiteralPath.ent_type
                    if with_type:
                        path.append(prefix)

                    path.append(f'{type_prefix}{next_ent_id}')
                    if dfs(uid, cur_ent_t, cand_type, cur_ent_id, next_ent_id, cur_hop+1, path, rel_id, n_hop, start_ent_type, end_ent_type,cur_attempts,max_attempts):
                        if with_type:
                            path.pop()
                        path.pop()
                        if with_type:
                            path.pop()
                        path.pop()                    
                        return True
                    else:
                        cur_attempts[0] += 1
                    if cur_attempts[0] >= max_attempts:
                        return True
                    if with_type:
                        path.pop()

                    path.pop()
                if with_type:
                    path.pop()
                path.pop()
        return False

    non_prod_entities = set([USER_ENT, EXT_ENT])
    user_products = list(user_dict[uid])
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
            cnt = 0
            
            while cnt < max_paths:

                if start_ent_type is None:
                    
                    cur_start_ent_type = random.choice([USER, PRODUCT, ENTITY])


                    if cur_start_ent_type == ENTITY:
                        non_ext_entity = set([USER, PRODUCT])
                        candidate_ext_ent_types = [x for x in kg if x not in non_ext_entity ]
                        cur_start_ent_type = random.choice(candidate_ext_ent_types)
                    
                    id = random.choice(list(kg[cur_start_ent_type]) ) 
                else:
                    cur_start_ent_type = start_ent_type
                    id = uid
                if n_hop is None:
                    valid_hop_range = []
                    if end_ent_type is  None:
                        valid_hop_range = [i for i in range(1,50+1)]
                    else:
                        
                        if (cur_start_ent_type in non_prod_entities and end_ent_type not in non_prod_entities) or \
                                (cur_start_ent_type not in non_prod_entities and end_ent_type in non_prod_entities) :
                            # only odd hops
                            valid_hop_range = [i for i in range(1,50+1, 2)]
                        else:
                            # only even hops
                            valid_hop_range = [i for i in range(2,50+1, 2)]
                         
                    cur_n_hop = random.choice(valid_hop_range)
                else:
                    cur_n_hop = n_hop

                

                path = []
                cur_hop = 0
                

                if cur_start_ent_type == USER_ENT:
                    if id == uid:
                        prefix = LiteralPath.main_user
                    else:
                        prefix = LiteralPath.oth_user
                    type_prefix = LiteralPath.user_type
                elif cur_start_ent_type == PROD_ENT:
                    if id in user_dict[uid]:  
                        prefix = LiteralPath.recom_prod
                    else:
                        prefix = LiteralPath.prod    
                    type_prefix = LiteralPath.prod_type                  
                else:              
                    prefix = LiteralPath.ent
                    type_prefix = LiteralPath.ent_type
                if with_type:
                    path.append(prefix)

                path.append(f'{type_prefix}{id}')     

                prev_ent_t = cur_start_ent_type
                cur_ent_t = cur_start_ent_type
                

                dfs(uid, cur_start_ent_type, cur_start_ent_type, id, id, cur_hop, path, -100, cur_n_hop, cur_start_ent_type, end_ent_type, [0])
                cnt += 1





class KGsampler:
    TOKEN_INDEX_FILE = 'token_index.txt'
    def __init__(self, dataset_name: str, save_dir='statistics', data_dir=None):
        path = get_data_dir(dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.dataset_info = KARSDataset(dataset_name, data_dir=data_dir)

        self.kg2t = KG_RELATION[dataset_name]
        self.token_index_filepath = os.path.join(path, KGsampler.TOKEN_INDEX_FILE)
        
        self.dataset_name = dataset_name
        print('Loading from ', path, ' the dataset ', dataset_name)
        item_list_file = os.path.join(path, 'i2kg_map.txt')#f'item_list.txt')
        kg_filepath = os.path.join(path,  f'kg_final.txt')                                      
        pid_mapping_filepath = os.path.join(path,  f'i2kg_map.txt')
        rel_mapping_filepath = os.path.join(path,  f'r_map.txt')
        rel_df = pd.read_csv(rel_mapping_filepath, sep='\t')
        pid_df = pd.read_csv(pid_mapping_filepath, sep='\t')
        
        self.pid2eid = { pid : eid for pid,eid in zip(pid_df.pid.values.tolist(), pid_df.eid.values.tolist())  }
        self.rel_id2type = { int(i) : rel_name for i,rel_name in zip(rel_df.id.values.tolist(), rel_df.name.values.tolist())  } 
        self.rel_id2type[int(LiteralPath.interaction_rel_id)] = INTERACTION[dataset_name]
        #print(self.rel_id2type)

        self.rel_type2id = { v:k for k,v in self.rel_id2type.items() }
        #print(self.rel_type2id)
        
        self.items = KGsampler.load_items(item_list_file)
        
        self.train_user_dict = self.load_user_inter(dataset_name, 'train')
        self.valid_user_dict = self.load_user_inter(dataset_name, 'valid')
        self.test_user_dict = self.load_user_inter(dataset_name, 'test')
        user_dict = defaultdict(set)
        for uid in self.train_user_dict:
            user_dict[uid].update(self.train_user_dict[uid])
            user_dict[uid].update(self.valid_user_dict[uid])
            user_dict[uid].update(self.test_user_dict[uid])  
        
        self.user_dict = user_dict
        # kg in h,r,t format
        self.kg, self.kg_np = KGsampler.load_kg(kg_filepath)
        self.graph_level_stats()
        #self.load_augmented_kg()
        self.load_augmented_kg_V2()
        
    def graph_level_stats(self):
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.n_relations = max(self.kg_np[:, 1]) + 1
        self.n_entities = max(max(self.kg_np[:, 0]), max(self.kg_np[:, 2])) + 1
        self.n_triples = len(self.kg_np)        
        
    def load_user_inter(self, dataset_name: str, set_name: str=None):
        G = defaultdict(set)
        curr_set = get_set(dataset_name, set_name)
        for uid, pids in curr_set.items():
            for pid in pids:
                G[uid].add(pid)
        return G        
    
    def load_items(item_file):
        item_ids = set()
        with open(item_file) as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                data = line.strip().rstrip().split('\t')
                #print(data)
                orig_id = int(data[1])
                item_id = int(data[0])
                item_ids.add(item_id)
        return item_ids
    
    def deg(self):
        degs = defaultdict(int)
        for h, rels in  self.kg.items():
            for r,tails in rels.items():
                for tail in tails:
                    degs[h] += 1
        return degs
    
    def load_kg(kg_filepath, undirected=True):
        kg = defaultdict()
        #kg_np = np.loadtxt(kg_filepath, np.uint32)
        kg_np = pd.read_csv(kg_filepath, sep='\t').to_numpy()
        print(kg_np.shape)
        kg_np = np.unique(kg_np, axis=0)
        print(kg_np.shape)
        for triple in kg_np:
            h,r,t = triple
            if h not in kg:
                kg[h] = defaultdict(set)
            if t not in kg:
                kg[t] = defaultdict(set)
            assert h != t, 'Self loop detected'
            kg[h][r].add(t)
            if undirected:
                kg[t][r].add(h)
            
        return kg, kg_np
    def build_token_index(self):

        aug_kg = self.aug_kg
        REL_TYPE2ID=self.rel_type2id
        kg_tokens = set()

        def get_token_ent_type(ent_type):
            token_type = None
            if ent_type == USER:
                token_type = LiteralPath.user_type
            elif ent_type == PRODUCT:
                token_type = LiteralPath.prod_type
            else:
                token_type = LiteralPath.ent_type
            return token_type

        for head_type in aug_kg:

            h_token_type = get_token_ent_type(head_type)

            for head_id in aug_kg[head_type]:
                head_token = f'{h_token_type}{head_id}'
                kg_tokens.add(head_token)
                for rel in aug_kg[head_type][head_id]:
                    rel_id = REL_TYPE2ID[rel]
                    rel_token = f'{LiteralPath.rel_type}{rel_id}'
                    kg_tokens.add(rel_token)
                    for tail_type in aug_kg[head_type][head_id][rel]:
                        t_token_type = get_token_ent_type(tail_type)
                        for tail_id in aug_kg[head_type][head_id][rel][tail_type]:
                            tail_token = f'{t_token_type}{tail_id}'
                            kg_tokens.add(tail_token)
        with open(self.token_index_filepath, 'w') as f:
            for token in kg_tokens:
                f.write(token + '\n')
                
        
    


    def random_walk_sampler(self, ignore_rels=set(), max_hop=None, max_paths=4000, logdir='paths_rand_walk',itemset_type='inner', 
        collaborative=True,
        nproc=8,
        with_type=True,
        start_ent_type=USER,
        end_ent_type=PRODUCT):
        user_dict, items = self.user_dict, self.items
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 

        func = random_walk_typified
        
        # undirected knowledge graph hypotesis (for each relation, there exists its inverse)

        #with mp.Pool(nproc) as pool:
        #    pool.starmap( functools.partial(func,   
        for uid in tqdm(list(self.user_dict) ) :       
            func(uid,                                                       
                dataset_name=self.dataset_name,
                kg=self.aug_kg, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, 
                                USER_ENT=USER, PROD_ENT=PROD_ENT, EXT_ENT=ENTITY, 
                                U2P_REL=U2P_REL,
                                logdir=os.path.join(self.save_dir, logdir),
                                user_dict=self.train_user_dict,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                REL_TYPE2ID=self.rel_type2id,
                                collaborative=collaborative,
                                dataset_info=self.dataset_info,
                                with_type=with_type,
                                start_ent_type=start_ent_type,
                                end_ent_type=end_ent_type)
        #    ,
        #        tqdm([[uid] for uid in self.user_dict ] ))

    
   
    def load_augmented_kg_V2(self):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        
        R2T = self.rel_id2type
        KG2T = KG_RELATION[self.dataset_name]
        print(R2T)
        
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        
        self.aug_kg = dict()
        self.aug_kg[USER] = dict()
        print('Creating augmented kg')
        for uid in user_dict:
            pids = user_dict[uid]
            self.aug_kg[USER][uid] = dict()
            self.aug_kg[USER][uid][U2P_REL] = defaultdict(list)
            
            if PROD_ENT not in self.aug_kg:
                self.aug_kg[PROD_ENT] = dict()
                
            for pid in pids:
                self.aug_kg[USER][uid][U2P_REL][PROD_ENT].append(pid)
                
                if pid not in self.aug_kg[PROD_ENT]:
                    self.aug_kg[PROD_ENT][pid] = dict()
                if U2P_REL not in self.aug_kg[PROD_ENT][pid]:
                    self.aug_kg[PROD_ENT][pid][U2P_REL] = defaultdict(list)
                    
                self.aug_kg[PROD_ENT][pid][U2P_REL][USER].append(uid)
        
        for h in self.kg:
            for rel, tails in self.kg[h].items():
                for t in tails:
                    # get tail entity type, uniquely determined by head_ent + rel_type
                    # kg is composed only of (h, REL, t)  where either of (h,t) can be PROD or EXTERNAL_ENT
                    TAIL_ENT = KG2T[PROD_ENT][R2T[rel]]
                    
                    h1,t1 = h,t
                    # to simplify the code, assume h is PROD, if it is not, swap it with the tail
                    if t in self.aug_kg[PROD_ENT]:
                        # swap them , to have product as head, just to reduce amount of code below
                        h1,t1 = t,h
                    #print(rel, h1, t1, '::::',h,t)
                    if h1 not in self.aug_kg[PROD_ENT]:
                        self.aug_kg[PROD_ENT][h1] = dict()
                    if R2T[rel] not in self.aug_kg[PROD_ENT][h1]:
                        self.aug_kg[PROD_ENT][h1][R2T[rel]] = defaultdict(list)
                    
                    if TAIL_ENT not in self.aug_kg:
                        self.aug_kg[TAIL_ENT] = dict()
                    if t1 not in self.aug_kg[TAIL_ENT]:
                        self.aug_kg[TAIL_ENT][t1] = dict()
                        
                    if R2T[rel] not in self.aug_kg[TAIL_ENT][t1]:
                        self.aug_kg[TAIL_ENT][t1][R2T[rel]] = defaultdict(list) 
                        
                    self.aug_kg[PROD_ENT][h1][R2T[rel]][TAIL_ENT].append( t1   )
                    self.aug_kg[TAIL_ENT][t1][R2T[rel]][PROD_ENT].append( h1   )
        print('Created augmented kg')              
        print('Creating token index')
        self.build_token_index()
        print('Created token index')




if __name__ == '__main__':
    MODEL = 'kgat'
    ML1M = 'ml1m'
    LFM1M ='lfm1m'
    CELL='cellphones'
    ROOT_DIR = os.environ('TREX_DATA_ROOT') if 'TREX_DATA_ROOT' in os.environ else '../..'
    # Dataset directories.
    DATA_DIR = {
        ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
        LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
        CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed/{MODEL}'
    }
    dataset_name = 'ml1m'
    dirpath = DATA_DIR[dataset_name]
    ml1m_kg = KGsampler(dirpath)
    dataset_name = 'lfm1m'
    dirpath = DATA_DIR[dataset_name]
    lfm1m_kg = KGsampler(dirpath)




