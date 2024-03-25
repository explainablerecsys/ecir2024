import csv
import gzip
import os
import pickle
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd
from pathlm.models.embeddings.kge_utils import get_dataset_info_dir
from pathlm.knowledge_graphs.kg_macros import ENTITY, PRODUCT, USER
from easydict import EasyDict as edict


class KARSDataset(object):
    """This class is used to load data files and save in the instance.
    This dataset is used for the following models: {PGPR, UCPR}
    """

    def __init__(self, dataset_name, set_name='train', data_dir=None):
        self.dataset_name = dataset_name
        if data_dir is None:
            self.data_dir = get_dataset_info_dir(self.dataset_name)
        else:
            self.data_dir = data_dir
        self.groupwise_global_eid_to_cat_eid = {}
        self.groupwise_global_eid_to_subtype = {}
        self.global_eid_to_cat_eid = {}
        self.cat_eid_to_global_eid = {}
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        entity_filename_edict, relation_filename_edict = self.infer_kg_structure()
        #Other relation names doesn't contain the main interaction
        self.entity_names, self.other_relation_names = list(entity_filename_edict.keys()), list(relation_filename_edict.keys())
        self.load_entities(entity_filename_edict)
        self.load_product_relations(relation_filename_edict)
        self.load_reviews()

    #This will not work if your entities are composed by multiple words, e.g. if you name an entity related_product
    #this script will consider as a relation, please use a single word for relations
    def infer_kg_structure(self):
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.txt.gz')]
        entity_filenames = [filename for filename in file_list if len(filename.split("_")) == 1]
        entity_filename_edict = edict()
        entity_names = []
        for entity_file in entity_filenames:
            if os.path.isdir(os.path.join(self.data_dir, entity_file)): continue
            name = entity_file.split(".")[0]
            if name in ["train", "valid", "test"]: continue
            entity_names.append(name)
            entity_filename_edict[name] = entity_file

        relation_filenames = [filename for filename in file_list if len(filename.split("_")) > 1]
        relation_filename_edict = edict()
        relation_names = []
        for relation_file in relation_filenames:
            name = relation_file.split(".")[0]
            relation_names.append(name)
            relation_filename_edict[name] = relation_file

        self.relation2entity = {}
        for rel_name in relation_names:
            entity_name = rel_name.split("_")[-1]
            self.relation2entity[rel_name] = entity_name

        return entity_filename_edict, relation_filename_edict

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self, entity_filename_edict):
        """Load 10 global entities from data files:
        'user','movie','actor','director','producer','production_company','category','editor','writter','cinematographer'
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """

        for name in entity_filename_edict:
            vocab = []
            self.global_eid_to_cat_eid[name] = dict()
            self.cat_eid_to_global_eid[name] = dict()
            # remove header with [1:]
            for x in self._load_file(entity_filename_edict[name])[1:]:
                cat_eid, global_eid = x.split("\t")
                vocab.append(cat_eid)
                global_eid = int(global_eid)
                cat_eid = int(cat_eid)
                group_id = None
                if name == PRODUCT:
                    global_eid = cat_eid
                    group_id = PRODUCT
                elif name == USER:
                    group_id = USER
                else:
                    group_id = ENTITY
                if group_id not in self.groupwise_global_eid_to_cat_eid:
                    self.groupwise_global_eid_to_cat_eid[group_id] = dict()
                    self.groupwise_global_eid_to_subtype[group_id] = dict()
                self.groupwise_global_eid_to_cat_eid[group_id][global_eid] = cat_eid
                self.groupwise_global_eid_to_subtype[group_id][global_eid] = name

                self.global_eid_to_cat_eid[name][global_eid] = cat_eid
                if name != PRODUCT and name != USER:
                    if ENTITY not in self.global_eid_to_cat_eid:
                        self.global_eid_to_cat_eid[ENTITY] = dict()
                    self.global_eid_to_cat_eid[ENTITY][global_eid] = cat_eid


                self.cat_eid_to_global_eid[name][cat_eid] = global_eid
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab) + 1))
            print('Load', name, 'of size', len(vocab))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `review_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, rating out of 5, timestamp)
        product_distrib = np.zeros(self.product.vocab_size)
        invalid_users = 0
        invalid_pid = 0

        self.users = set()
        self.products = set()
        self.user_pids = defaultdict(set)
        self.pid_users = defaultdict(set)
        self.user_negative_pids = dict()
        self.pid_negative_users = dict()

        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            rating = int(arr[2])
            timestamp = int(arr[3])
            review_data.append((user_idx, product_idx, rating, timestamp))
            self.user_pids[user_idx].add(product_idx)
            self.pid_users[product_idx].add(user_idx)
            self.users.add(user_idx)
            self.products.add(product_idx)

            product_distrib[product_idx] += 1

        for uid in self.users:
            self.user_negative_pids[uid] = list(self.products - self.user_pids[uid])
        for pid in self.products:
            self.pid_negative_users[pid] = list(self.users - self.pid_users[pid])


        print(f"Invalid users: {invalid_users}, invalid items: {invalid_pid}")
        self.review = edict(
            data=review_data,
            size=len(review_data),
            product_distrib=product_distrib,
            product_uniform_distrib=np.ones(self.product.vocab_size),
            review_count=len(review_data),
            review_distrib=np.ones(len(review_data))  # set to 1 now
        )



        print('Load review of size', self.review.size)

    def load_product_relations(self, relation_filename_edict):
        """Load 8 product -> ? relations:
        - 'directed_by': movie -> director
        - 'produced_by_company': movie->production_company,
        - 'produced_by_producer': movie->producer,
        - 'starring': movie->actor,
        - 'belong_to': movie->category,
        - 'edited_by': movie->editor,
        - 'written_by': movie->writter,
        - 'cinematography': movie->cinematographer,

        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """

        product_relations = edict()
        for rel_name, rel_filename in relation_filename_edict.items():
            entity_name = self.relation2entity[rel_name]
            product_relations[rel_name] = (rel_filename, getattr(self, entity_name))
        #E.g:
        #    product_relations = edict(
        #        belong_to=("belong_to_genre.txt.gz", self.genre),
        #        featured_by=("featured_by_artist.txt.gz", self.artist),
        #        mixed_by=('mixed_by_engineer.txt.gz', self.engineer),
        #        produced_by=('produced_by_producer.txt.gz', self.producer),
        #    )


        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                data=[],
                et_vocab=product_relations[name][1].vocab,  # copy of brand, catgory ... 's vocab
                et_distrib=np.zeros(product_relations[name][1].vocab_size)  # [1] means self.brand ..
            )
            size = 0
            for line in self._load_file(product_relations[name][0]):  # [0] means brand_p_b.txt.gz ..
                knowledge = []
                line = line.split('\t')
                for x in line:  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                        size += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', size)

        all_kg_entities = set()
        pid_to_kg_entities_negative_samples = dict()
        pid_to_kg_entities = defaultdict(set)
        kg_entities_to_pid = defaultdict(set)
        kg_entities_to_pid_negative_samples = dict()
        #kg_entities_to_ent_negative_samples = defaultdict(set)

        self.unique_relations = set()
        for rel_name in product_relations:
            self.unique_relations.add(rel_name)
            relation_edict = getattr(self, rel_name)
            for pid, kg_entities in enumerate(relation_edict.data):
                pid_to_kg_entities[(pid, rel_name)] = set(kg_entities)
                for kg_ent in kg_entities:
                    kg_entities_to_pid[(kg_ent, rel_name)].add(pid)
                all_kg_entities.update(kg_entities)

        for (pid, rel_name), kg_entities in pid_to_kg_entities.items():
            pid_to_kg_entities_negative_samples[(pid, rel_name)] = list(all_kg_entities - kg_entities)

        products = set( [ k1 for k1,k2 in  pid_to_kg_entities.keys()] )
        for (kg_ent, rel_name), pids in kg_entities_to_pid.items():
            kg_entities_to_pid_negative_samples[(kg_ent, rel_name)] = list(products - pids)


        self.pid_to_kg_entities = pid_to_kg_entities
        self.pid_to_kg_entities_negative_samples = pid_to_kg_entities_negative_samples
        self.all_kg_entities = all_kg_entities
        self.kg_entities_to_pid = kg_entities_to_pid
        self.kg_entities_to_pid_negative_samples = kg_entities_to_pid_negative_samples

    def get_path_types_in_kg(self) -> Set[str]:
        return self.unique_relations

    def get_no_path_types_in_kg(self) -> int:
        return len(self.unique_relations)

    def create_or_load_linking_interaction_recency_matrix(self):
        lir_matrix_filepath = os.path.join(self.data_dir, "LIR_matrix.pkl")

        if os.path.isfile(lir_matrix_filepath):
            print("Loading pre-computed LIR-matrix")
            with open(lir_matrix_filepath, 'rb') as f:
                self.LIR_matrix = pickle.load(f)
            f.close()
        else:
            print("Generating LIR-matrix")
            self.create_linking_interaction_recency_matrix()
            with open(lir_matrix_filepath, 'wb') as f:
                pickle.dump(self.LIR_matrix, f)
            f.close()

    def create_linking_interaction_recency_matrix(self):
        self.get_interaction2timestamp_map()
        self.LIR_matrix = {uid: {} for uid in self.uid_timestamp.keys()}
        for uid in self.uid_timestamp.keys():
            linked_interaction_types = [PRODUCT]
            for linked_entity_type in linked_interaction_types:
                interactions = [type_id_time for type_id_time in self.uid_timestamp[uid] if
                                type_id_time[0] == linked_entity_type]
                interactions.sort(key=lambda x: x[2])
                if len(self.uid_timestamp[uid]) <= 1:  # Skips users with only one review in train (can happen with lastfm)
                    continue
                ema_timestamps = normalized_ema([x[2] for x in interactions])
                pid_lir = {}
                for i in range(len(interactions)):
                    pid = interactions[i][1]
                    lir = ema_timestamps[i]
                    pid_lir[pid] = lir
                self.LIR_matrix[uid][linked_entity_type] = pid_lir

    def get_interaction2timestamp_map(self):
        # Load if already computated
        metadata_filepath = os.path.join(self.data_dir, "time_metadata.pkl")
        if os.path.isfile(metadata_filepath):
            with open(metadata_filepath, 'rb') as f:
                user2pid_time_tuple = pickle.load(f)
            f.close()
            return user2pid_time_tuple
        # Compute and save if not yet
        user2pid_time_tuple = defaultdict(list)
        file = open(os.path.join(self.data_dir, "train.txt"), 'r')
        csv_reader = csv.reader(file, delimiter='\t')
        #uid_mapping = get_dataset_id2model_kg_id(dataset_name, model_name, "user")
        for row in csv_reader:
            uid = row[0] #TODO: Check if removal of uid_mapping give problems
            pid = row[1]
            pid_model_kg = self.pid_to_kg_entities[pid]
            timestamp = int(row[3])
            user2pid_time_tuple[uid].append((PRODUCT, pid_model_kg, timestamp))
        with open(os.path.join(metadata_filepath), 'wb') as f:
            pickle.dump((user2pid_time_tuple), f)
        f.close()
        return user2pid_time_tuple


    def create_or_load_shared_entity_popularity_matrix(self):
        sep_matrix_filepath = os.path.join(self.data_dir, "SEP_matrix.pkl")
        if os.path.isfile(sep_matrix_filepath):
            print("Loading pre-computed SEP-matrix")
            with open(sep_matrix_filepath, 'rb') as f:
                self.SEP_matrix = pickle.load(f)
            f.close()
        else:
            print("Generating SEP-matrix")
            self.SEP_matrix = self.create_shared_entity_popularity_matrix()
            with open(sep_matrix_filepath, 'wb') as f:
                pickle.dump(self.SEP_matrix, f)
            f.close()

    def create_shared_entity_popularity_matrix(self):
        # Precompute entity distribution
        SEP_matrix = {}
        degrees = load_kg(dataset_name, model_name).degrees
        for type, eid_degree in degrees.items():
            eid_degree_tuples = list(zip(eid_degree.keys(), eid_degree.values()))
            eid_degree_tuples.sort(key=lambda x: x[1])
            ema_es = normalized_ema([x[1] for x in eid_degree_tuples])
            pid_weigth = {}
            for idx in range(len(ema_es)):
                pid = eid_degree_tuples[idx][0]
                pid_weigth[pid] = ema_es[idx]

            SEP_matrix[type] = pid_weigth

        return SEP_matrix

def normalized_ema(values):
    if max(values) == min(values):
        values = np.array([i for i in range(len(values))])
    else:
        values = np.array([i for i in values])
    values = pd.Series(values)
    ema_vals = values.ewm(span=len(values)).mean().tolist()
    min_res = min(ema_vals)
    max_res = max(ema_vals)
    return [(x - min_res) / (max_res - min_res) for x in ema_vals]