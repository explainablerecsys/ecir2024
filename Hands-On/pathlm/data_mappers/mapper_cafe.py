import numpy as np
import os
import gzip
import csv
import pandas as pd

from pathlm.knowledge_graphs.kg_macros import PRODUCT, INTERACTION
from pathlm.utils import get_data_dir, check_dir, get_model_data_dir

from pathlm.data_mappers.mapper_base import MapperBase


class MapperCAFE(MapperBase):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = args.data
        self.input_folder = get_data_dir(self.dataset_name)
        self.output_folder = get_model_data_dir(self.model_name, self.dataset_name)
        check_dir(self.output_folder)
        print("Mapping to CAFE...")
        self.map_to_CAFE()
        print("Getting splits...")
        self.get_splits()
        print("Writing split CAFE...")
        self.write_split_CAFE()
        print("Writing UID and PID mappings...")
        self.mapping_folder = os.path.join(get_data_dir(args.data), "mapping")
        check_dir(self.mapping_folder)
        self.write_uid_pid_mappings()

    def write_split_CAFE(self):
        for set in ["train", "valid", "test"]:
            with gzip.open(os.path.join(self.output_folder, f"{set}.txt.gz"), 'wt') as set_file:
                writer = csv.writer(set_file, delimiter="\t")
                set_values = getattr(self, set)
                for uid in set_values.keys():
                    pid_time_tuples = set_values[uid]
                    uid = self.ratings_uid2new_id[uid]
                    pids = [self.ratings_pid2new_id[pid] for pid, time in pid_time_tuples]
                    writer.writerow([uid] + pids)
            set_file.close()


    def map_to_CAFE(self):
        relations_df = pd.read_csv(os.path.join(self.input_folder, "r_map.txt"), sep="\t")
        self.rid2entity_name = dict(zip(relations_df.id, relations_df.name.apply(lambda x: x.split("_")[-1])))
        self.rid2relation_name = dict(zip(relations_df.id, relations_df.name))

        self.handle_kg_relations()
        self.handle_kg_rules()
        self.handle_kg_entities()
        self.collect_triplets()

    def handle_kg_relations(self):
        # Write new relations
        new_rid2relation_name = {0: INTERACTION[self.dataset_name]}
        offset = len(new_rid2relation_name)
        new_rid2relation_name.update({rid + offset: relation_name for rid, relation_name in self.rid2relation_name.items()})
        self.old_rid2new_rid = {rid: rid + offset for rid in self.rid2relation_name.keys()}

        # Add reverse relations
        n_relations = len(new_rid2relation_name)
        new_rid2relation_name.update(
            {n_relations + rid: f"rev_{relation_name}" for rid, relation_name in new_rid2relation_name.items()})

        # Save kg_relations file
        relations_df = pd.DataFrame(list(new_rid2relation_name.items()), columns=["id", "name"])
        relations_df.to_csv(os.path.join(self.output_folder, "kg_relations.txt.gz"), index=False, header=False, sep="\t",
                            compression="gzip")

    def handle_kg_rules(self):
        main_relation = 0
        n_relations = len(self.rid2relation_name)+1
        self.rid2reverse_rid = {rid: n_relations + rid for rid in range(n_relations)}

        with gzip.open(os.path.join(self.output_folder, 'kg_rules.txt.gz'), 'wt') as kg_rules_file:
            writer = csv.writer(kg_rules_file, delimiter="\t")
            writer.writerow(
                [main_relation, self.rid2reverse_rid[main_relation], main_relation])  # Special case for main interaction
            for rid in range(1, n_relations):  # Start from 1 to skip the main interaction
                forward, reverse = rid, self.rid2reverse_rid[rid]
                writer.writerow([main_relation, forward, reverse])
    def handle_kg_entities(self):
        e_map_df = self._load_e_map()
        entity2old_eid = dict(zip(e_map_df.entity, e_map_df.old_eid))
        old_eid2entity = dict(zip(e_map_df.old_eid.astype(int), e_map_df.entity))


        self.all_entities_df = pd.DataFrame([], columns=["local_id", "name"])

        # Collecting user entities
        self.all_entities_df = self._collect_user_entities()

        # Collecting product entities
        self.all_entities_df = self._collect_product_entities(entity2old_eid)

        # Collecting external entities
        self.all_entities_df = self._collect_external_entities(old_eid2entity)

        # Write kg_entities.txt.gz
        self._write_kg_entities()
        self._extract_ratings_to_global_id()

    def _extract_ratings_to_global_id(self):
        # Extract ratings id (uid) to global id
        user_entities_df = self.all_entities_df[self.all_entities_df.local_id.str.contains("user")].copy()
        user_entities_df.loc[:, 'name'] = user_entities_df['name'].apply(lambda x: x.split("_")[-1])
        self.rating_uid2global_id = dict(zip(user_entities_df.name, user_entities_df.global_id))

        # Extract ratings id (pid) to global id
        item2kg_df = pd.read_csv(os.path.join(self.input_folder, "i2kg_map.txt"), sep="\t")
        item2kg_df = item2kg_df[["entity", "pid"]]
        item2kg_df = pd.merge(item2kg_df, self.all_entities_df[self.all_entities_df.local_id.str.contains("product_")],
                              left_on="entity", right_on="name")
        self.rating_pid2global_id = dict(zip(item2kg_df.pid.astype(str), item2kg_df.global_id))

    def _load_e_map(self):
        return pd.read_csv(os.path.join(self.input_folder, "e_map.txt"), sep="\t", names=["old_eid", "name", "entity"]).iloc[1:, ][
            ["old_eid", "entity"]]

    def _collect_user_entities(self):
        user_df = pd.read_csv(os.path.join(self.input_folder, "users.txt"), sep="\t")
        user_df["local_id"] = "user_" + user_df.index.astype(str)
        self.ratings_uid2new_id = dict(zip(user_df.uid.astype(str), user_df.index.astype(str)))
        user_df["name"] = "user_" + user_df.uid.astype(str)
        user_df["old_eid"] = np.nan
        return pd.concat([self.all_entities_df, user_df[["local_id", "name"]]], ignore_index=True)

    def _collect_product_entities(self, entity2old_eid):
        pid2kg_df = pd.read_csv(os.path.join(self.input_folder, "i2kg_map.txt"), sep="\t")
        pid2kg_df["local_id"] = "product_" + pid2kg_df.index.astype(str)
        self.ratings_pid2new_id = dict(zip(pid2kg_df.pid.astype(str), pid2kg_df.index.astype(str)))
        pid2kg_df["name"] = pid2kg_df.entity
        pid2kg_df["old_eid"] = pid2kg_df.entity.astype(str).map(entity2old_eid)
        assert pid2kg_df.old_eid.isnull().sum() == 0
        return pd.concat([self.all_entities_df, pid2kg_df[["local_id", "name", "old_eid"]]], ignore_index=True)

    def _collect_external_entities(self, old_eid2entity):
        self.kg_df = pd.read_csv(os.path.join(self.input_folder, "kg_final.txt"), sep="\t")
        assert self.kg_df.entity_tail.isnull().sum() == 0
        assert self.kg_df.entity_head.isnull().sum() == 0
        for rid, entity_name in self.rid2entity_name.items():
            unique_entities_by_type = self.kg_df[self.kg_df.relation == rid].entity_tail.unique()
            entity_by_type_df = pd.DataFrame({"old_eid": unique_entities_by_type})
            entity_by_type_df["local_id"] = entity_name + "_" + entity_by_type_df.index.astype(str)
            entity_by_type_df["name"] = entity_by_type_df.old_eid.map(old_eid2entity)
            self.all_entities_df = pd.concat([self.all_entities_df, entity_by_type_df[["local_id", "name", "old_eid"]]], ignore_index=True)
        return self.all_entities_df

    def _write_kg_entities(self):
        self.all_entities_df["global_id"] = self.all_entities_df.index
        all_entities_df = self.all_entities_df[["global_id", "local_id", "name"]]
        all_entities_df.to_csv(os.path.join(self.output_folder, "kg_entities.txt.gz"), index=False, sep="\t",
                               compression="gzip")

    def collect_triplets(self):
        # Initialize DataFrame for triplets
        triplets_df = pd.DataFrame(columns=["entity_head", "relation", "entity_tail"])

        # Collect user interaction triplets
        self.get_splits()
        interaction_triplets = self._collect_interaction_triplets()
        interaction_df = pd.DataFrame(interaction_triplets, columns=["entity_head", "relation", "entity_tail"], dtype="int64")

        # Map old entity IDs to global IDs
        self.old_eid2global_id = self._map_old_eid_to_global_id()

        # Insert other entities to kg_triplets df
        other_triplets_df = self._insert_other_entities_to_triplets_df(triplets_df)

        # Concatenate the DataFrames
        triplets_df = pd.concat([triplets_df, interaction_df, other_triplets_df], ignore_index=True)

        # Save triplets DataFrame
        output_folder = get_model_data_dir(self.model_name, self.dataset_name)
        triplets_df.to_csv(os.path.join(output_folder, "kg_triples.txt.gz"), index=False, sep="\t", compression="gzip")

    def _collect_interaction_triplets(self):
        interaction_triplets = []
        main_interaction, rev_main_interaction = 0, self.rid2reverse_rid[0]
        for uid, pid_time_tuples in self.train.items():
            uid_global = self.rating_uid2global_id[uid]
            for pid, _ in pid_time_tuples:
                pid_global = self.rating_pid2global_id[pid]
                interaction_triplets.extend(
                    [[uid_global, main_interaction, pid_global], [pid_global, rev_main_interaction, uid_global]])
        return interaction_triplets

    def _map_old_eid_to_global_id(self):
        external_entities_df = self.all_entities_df[~self.all_entities_df.local_id.str.contains("user")]
        old_eid2global_id = {
            PRODUCT: dict(
                zip(external_entities_df[external_entities_df.local_id.str.contains("product")].old_eid.astype(int),
                    external_entities_df.global_id))
        }
        for rid, entity_name in self.rid2entity_name.items():
            entities_by_type = self.kg_df[self.kg_df.relation == rid].entity_tail.unique()
            entity_by_type_df = pd.DataFrame({"old_eid": entities_by_type.astype(int)})
            entity_by_type_df = entity_by_type_df.merge(external_entities_df, on="old_eid", how="left").dropna()
            old_eid2global_id[entity_name] = dict(zip(entity_by_type_df.old_eid, entity_by_type_df.global_id))
        return old_eid2global_id

    def _insert_other_entities_to_triplets_df(self, triplets_df):
        all_triplets = [triplets_df]  # Start with the existing triplets
        for rid, entity_name in self.rid2entity_name.items():
            triplets_by_type = self.kg_df[self.kg_df.relation == rid].copy()
            triplets_by_type.entity_head = triplets_by_type.entity_head.map(self.old_eid2global_id[PRODUCT])
            triplets_by_type.entity_tail = triplets_by_type.entity_tail.map(self.old_eid2global_id[entity_name])
            triplets_by_type.relation = triplets_by_type.relation.map(self.old_rid2new_rid)

            # Debug: Print NaN values count
            #print(f"NaN values in entity_head: {triplets_by_type.entity_head.isna().sum()}")
            #print(f"NaN values in entity_tail: {triplets_by_type.entity_tail.isna().sum()}")

            # Filter out rows with NaN values
            triplets_by_type.dropna(subset=['entity_head', 'entity_tail'], inplace=True)

            rev_triplets_by_type = triplets_by_type.rename(
                columns={"entity_head": "entity_tail", "entity_tail": "entity_head"})
            rev_triplets_by_type.relation = rev_triplets_by_type.relation.map(self.rid2reverse_rid)

            all_triplets.extend([triplets_by_type, rev_triplets_by_type])

        # Concatenate all triplets
        return pd.concat(all_triplets, ignore_index=True)