import csv
import os
import pandas as pd
from pathlm.utils import get_data_dir

class MapperBase(object):
    def __init__(self,args):
        self.args = args
        self.dataset_name = args.data
        self.model_name = args.model
        self.valid_size = args.valid_size
        self.train_size = args.train_size
        self.test_size = 1 - args.train_size - args.valid_size
        print(f"Creating data/{self.dataset_name}/preprocessed/{self.model_name}/ filesystem")

    def get_splits(self):
        input_dir = get_data_dir(self.dataset_name)
        self.train, self.valid, self.test = {}, {}, {}
        for set in ["train", "valid", "test"]:
            with open(os.path.join(input_dir, f"{set}.txt"), 'r') as set_file:
                curr_set = getattr(self, set)
                reader = csv.reader(set_file, delimiter="\t")
                for row in reader:
                    uid, pid, interaction, time = row
                    if uid not in curr_set:
                        curr_set[uid] = []
                    curr_set[uid].append((pid, time))
            set_file.close()

    def write_uid_pid_mappings(self):
        ratings_uid2new_id_df = pd.DataFrame(list(zip(self.ratings_uid2new_id.values(), self.ratings_uid2new_id.keys())),
                                             columns=["new_id", "rating_id"])
        ratings_uid2new_id_df.to_csv(os.path.join(self.mapping_folder, "user.txt"), sep="\t", index=False)

        ratings_pid2new_id_df = pd.DataFrame(list(zip(self.ratings_pid2new_id.values(), self.ratings_pid2new_id.keys())),
                                             columns=["new_id", "rating_id"])
        ratings_pid2new_id_df.to_csv(os.path.join(self.mapping_folder, "product.txt"), sep="\t", index=False)