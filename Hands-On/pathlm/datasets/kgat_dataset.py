import collections
import numpy as np
import os
from torch.utils.data import Dataset

class KGATStyleDataset(Dataset):
    """
    This dataset is used by the following models: {BPRMF, FM, NFM, CKE, CFKG, KGAT}
    """
    def __init__(self, args, path, batch_style='list'):
        super(KGATStyleDataset).__init__()

        self.batch_styles = {'list': 0, 'map': 1}
        self.mode = 'cf'
        assert batch_style in self.batch_styles, f"Error: got {batch_style} but valid batch styles are {list(self.batch_styles.keys())}"
        self.path = path
        self.args = args
        self.batch_style = batch_style
        self.batch_style_id = self.batch_styles[self.batch_style]
        self.batch_size = args.batch_size

        # Load data
        self.products = self._load_products(os.path.join(path, 'item_list.txt'))
        self.train_data, self.train_user_dict = self._load_ratings(os.path.join(path, 'train.txt'))
        self.valid_data, self.valid_user_dict = self._load_ratings(os.path.join(path, 'valid.txt'))
        self.test_data, self.test_user_dict = self._load_ratings(os.path.join(path, 'test.txt'))

        self.exist_users = list(self.train_user_dict.keys())
        self.N_exist_users = len(self.exist_users)

        self._statistic_ratings()

        # Load KG data
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(os.path.join(path, 'kg_final.txt'))

        # Print dataset info
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size)
        self._print_data_info()

        self.layer_size = eval(args.layer_size)[0]
        self.mess_dropout = eval(args.mess_dropout)[0]
        self.node_dropout = eval(args.mess_dropout)[0]

    def _load_ratings(self, file_name):
        user_dict = collections.defaultdict(list)
        inter_mat = []

        with open(file_name, 'r') as file:
            for line in file:
                items = [int(i) for i in line.strip().split()]
                user, pos_items = items[0], items[1:]
                inter_mat.extend([(user, item) for item in pos_items])
                user_dict[user].extend(pos_items)

        return np.array(inter_mat), user_dict

    def _load_products(self, file_name):
        products = set()
        with open(file_name, 'r') as file:
            file.readline()
            for line in file:
                eid_product = int(line.split(' ')[1])
                products.add(eid_product)
        return products

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.valid_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)

    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)
            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print(f'[n_users, n_items]=[{self.n_users}, {self.n_items}]')
        print(f'[n_train, n_test]=[{self.n_train}, {self.n_test}]')
        print(f'[n_entities, n_relations, n_triples]=[{self.n_entities}, {self.n_relations}, {self.n_triples}]')
        print(f'[batch_size, batch_size_kg]=[{self.batch_size}, {self.batch_size_kg}]')
