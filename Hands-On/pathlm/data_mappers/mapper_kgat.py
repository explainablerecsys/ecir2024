import argparse
import os
import csv
import pandas as pd
from typing import Dict, List, Any

from pathlm.utils import get_data_dir, get_model_data_dir, check_dir

from pathlm.data_mappers.mapper_base import MapperBase


class MapperKGAT(MapperBase):
    def __init__(self, args: Any):
        super().__init__(args)
        self.sep = " "
        self.input_folder = get_data_dir(self.dataset_name)
        self.output_folder = get_model_data_dir(self.model_name, self.dataset_name)
        check_dir(self.output_folder)
        self.mapping_folder = os.path.join(self.output_folder, "mappings")
        check_dir(self.mapping_folder)

        print("Mapping to KGAT...")
        self.map_to_KGAT()

        print("Getting data splits...")
        self.get_splits()

        print("Writing split data for KGAT...")
        self.write_split_KGAT()

        print("Writing UID and PID mappings...")
        self.write_uid_pid_mappings()

    def map_to_KGAT(self) -> None:
        check_dir(self.output_folder)

        self._map_items()
        self._map_entities()
        self._map_users()
        self._map_relations()
        self._map_kg_final()

    def _map_items(self) -> None:
        i2kg_df = pd.read_csv(os.path.join(self.input_folder, "i2kg_map.txt"), sep="\t")[["entity", "pid"]]
        i2kg_df.insert(1, "remap_id", list(range(i2kg_df.shape[0])))
        i2kg_df.rename(columns={"pid": "org_id"}, inplace=True)
        i2kg_df = i2kg_df[["org_id", "remap_id", "entity"]]
        i2kg_df.to_csv(os.path.join(self.output_folder, "item_list.txt"), sep=self.sep, index=False)
        self.ratings_pid2new_id: Dict[int, int] = dict(zip(i2kg_df.org_id, i2kg_df.remap_id))

    def _map_entities(self) -> None:
        entity_df = \
        pd.read_csv(os.path.join(self.input_folder, "e_map.txt"), sep="\t")[
            ["entity", "eid"]]
        entity_df.rename(columns={"entity": "org_id", "eid": "remap_id"}, inplace=True)
        entity_df.to_csv(os.path.join(self.output_folder, "entity_list.txt"), sep=self.sep, index=False)

    def _map_users(self) -> None:
        users_df = pd.read_csv(os.path.join(self.input_folder, "users.txt"), sep="\t")
        users_df.insert(1, "remap_id", list(range(users_df.shape[0])))
        users_df.rename(columns={"uid": "org_id"}, inplace=True)
        users_df.to_csv(os.path.join(self.output_folder, "user_list.txt"), sep=self.sep, index=False)
        self.ratings_uid2new_id: Dict[int, int] = dict(zip(users_df.org_id, users_df.remap_id))

    def _map_relations(self) -> None:
        r_map_df = pd.read_csv(os.path.join(self.input_folder, "r_map.txt"), sep="\t")
        r_map_df.rename(columns={"id": "remap_id", "kb_relation": "org_id"}, inplace=True)
        r_map_df = r_map_df[["org_id", "remap_id"]]
        r_map_df.to_csv(os.path.join(self.output_folder, "relation_list.txt"), sep=self.sep, index=False)

    def _map_kg_final(self) -> None:
        kg_final_df = pd.read_csv(os.path.join(self.input_folder, "kg_final.txt"), sep="\t")[
            ["entity_head", "relation", "entity_tail"]]
        kg_final_df.to_csv(os.path.join(self.output_folder, "kg_final.txt"), sep=self.sep, header=False, index=False)

    def write_split_KGAT(self) -> None:
        for set_name in ["train", "valid", "test"]:
            with open(os.path.join(self.output_folder, f"{set_name}.txt"), 'w') as set_file:
                writer = csv.writer(set_file, delimiter=self.sep)
                set_values = getattr(self, set_name)
                for uid, pid_time_tuples in set_values.items():
                    uid = self.ratings_uid2new_id[int(uid)]
                    pids = [self.ratings_pid2new_id[int(pid)] for pid, _ in pid_time_tuples]
                    writer.writerow([uid] + pids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ml1m', help='One of {ML1M, LFM1M}')
    parser.add_argument('--model', type=str, default='kgat', help='')
    parser.add_argument('--train_size', type=float, default=0.6, help='size of the train set expressed in 0.x')
    parser.add_argument('--valid_size', type=float, default=0.2, help='size of the valid set expressed in 0.x')
    args = parser.parse_args()

    MapperKGAT(args)

if __name__ == '__main__':
    main()