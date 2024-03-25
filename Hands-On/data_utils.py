import csv
import os
from collections import defaultdict, Counter

import pandas as pd
from utils import get_data_dir, get_raw_data_dir, ML1M, LFM1M


def time_based_train_test_split(dataset_name, train_size, valid_size):
    dataset_name = dataset_name
    input_folder = get_data_dir(dataset_name)
    output_folder = input_folder

    uid2pids_timestamp_tuple = defaultdict(list)
    with open(os.path.join(input_folder, 'ratings.txt'), 'r') as ratings_file:  # uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[uid].append([pid, int(timestamp)])
    ratings_file.close()

    for uid in uid2pids_timestamp_tuple.keys():
        uid2pids_timestamp_tuple[uid].sort(key=lambda x: x[1])

    train, valid, test = {}, {}, {}
    for uid, pid_time_tuples in uid2pids_timestamp_tuple.items():
        n_interactions = len(pid_time_tuples)
        train_end = int(n_interactions * train_size)
        valid_end = train_end + int(n_interactions * valid_size)+1
        train[uid], valid[uid], test[uid] = pid_time_tuples[:train_end], pid_time_tuples[train_end:valid_end], pid_time_tuples[valid_end:]

    for set_filename in [(train, "train.txt"), (valid, "valid.txt"), (test, "test.txt")]:
        set_values, filename = set_filename
        with open(os.path.join(output_folder, filename), 'w') as set_file:
            writer = csv.writer(set_file, delimiter="\t")
            for uid, pid_time_tuples in set_values.items():
                for pid, time in pid_time_tuples:
                    writer.writerow([uid, pid, 1, time])
        set_file.close()

def add_products_metadata(dataset_name):
    raw_data_dir = get_raw_data_dir(dataset_name)
    data_dir = get_data_dir(dataset_name)
    products_df = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    if dataset_name == ML1M:
        #Add provider
        provider_df = pd.read_csv(os.path.join(raw_data_dir, "directions.dat"), sep="::")
        provider_df = provider_df.astype("object")
        provider_df = provider_df[provider_df.movieId.isin(products_df.pid)]
        provider_df.drop_duplicates(subset="movieId", inplace=True)
        products_df = pd.merge(products_df, provider_df, how="outer", left_on="pid", right_on="movieId")
        if products_df.dirId.isnull().values.any():
            products_df.dirId.fillna(-1, inplace=True)
        products_df.rename({"movie_name": "name", "dirId": "provider_id"}, axis=1, inplace=True)
    elif dataset_name == LFM1M:
        products_df.rename({"artist_id": "provider_id"}, axis=1, inplace=True)

    #Add item popularity
    interactions_df = pd.read_csv(os.path.join(data_dir, "train.txt"), sep="\t", names=["uid", "pid", "interaction", "timestamp"])
    product2interaction_number = Counter(interactions_df.pid)
    most_interacted = max(product2interaction_number.values())
    less_interacted = 0 if len(list(product2interaction_number.keys())) != products_df.pid.unique().shape[0] \
        else min(product2interaction_number.values())
    for pid in list(products_df.pid.unique()):
        occ = product2interaction_number[pid] if pid in product2interaction_number else 0
        product2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)

    products_df.insert(3, "pop_item", product2interaction_number.values(), allow_duplicates=True)

    #Add provider popularity
    item2provider = dict(zip(products_df.pid, products_df.provider_id))
    interaction_provider_df = interactions_df.copy()
    interaction_provider_df['provider_id'] = interaction_provider_df.pid.map(item2provider)
    provider2interaction_number = Counter(interaction_provider_df.provider_id)
    provider2interaction_number[-1] = 0
    most_interacted, less_interacted = max(provider2interaction_number.values()), min(provider2interaction_number.values())
    for pid in provider2interaction_number.keys():
        occ = provider2interaction_number[pid]
        provider2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)
    products_df["pop_provider"] = products_df.provider_id.map(provider2interaction_number)
    products_df = products_df[["pid", "name", "provider_id", "genre", "pop_item", "pop_provider"]]
    products_df.to_csv(os.path.join(data_dir, "products.txt"), sep="\t", index=False)

def categorize_uses_age(dataset_name):
    data_dir = get_data_dir(dataset_name)
    users_df = pd.read_csv(os.path.join(data_dir, "users.txt"), sep="\t")
    users_df.age = users_df.age.apply(lambda x: categorigal_to_categorigal_age(x))
    users_df.to_csv(f"{data_dir}/users.txt", sep="\t", index=False)

def categorigal_to_categorigal_age(age_group_number):
    if age_group_number == 1:   return "Under 18"
    if age_group_number == 18:  return "18-24"
    if age_group_number == 25:  return "25-34"
    if age_group_number == 35:  return "35-44"
    if age_group_number == 45:  return "45-49"
    if age_group_number == 50:  return "50-55"
    if age_group_number == 56:  return "56+"