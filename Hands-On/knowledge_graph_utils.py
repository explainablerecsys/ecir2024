
from typing import Tuple
import csv

import pandas as pd

from utils import get_dataframe_from_json


def propagate_item_removal_to_kg(items_df: pd.DataFrame, items_to_kg_df: pd.DataFrame, entities_df: pd.DataFrame, kg_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Updates i2kg, e_map and kg_df based on the products in items_df, returns this 3 updated dataframes
    """
    items_to_kg_df_after = items_to_kg_df[items_to_kg_df.dataset_id.isin(items_df.movie_id)]
    removed_items = items_to_kg_df[~items_to_kg_df.dataset_id.isin(items_to_kg_df_after.dataset_id)]
    print(f"Removed {removed_items.shape[0]} entries from i2kg map.")
    removed_entities = entities_df[entities_df.entity_url.isin(removed_items.entity_url)]
    print(f"Removed {removed_entities.shape[0]} entries from e_map")
    entities_df = entities_df[~entities_df.entity_url.isin(removed_items.entity_url)]
    n_triplets = kg_df.shape[0]
    kg_df = kg_df[~kg_df.entity_head.isin(removed_entities.entity_id)]
    print(f"Removed {n_triplets - kg_df.shape[0]} triplets from kg_df")
    return items_to_kg_df_after, entities_df, kg_df

def create_kg_from_metadata(dataset: str) -> None:
    """
    Creates the KG standard format file from the meta_Cell_Phones_and_Accessories, it can be used to convert other
    amazon datasets you just need to change the meta filepath
    """
    input_data = f'data/{dataset}/preprocessed'
    input_kg = f'data/{dataset}/kg'
    metaproduct_df = get_dataframe_from_json(input_kg + '/meta_Cell_Phones_and_Accessories.json.gz')
    metaproduct_df = metaproduct_df.drop(['tech1', 'description', 'fit', 'title', 'tech2', 'feature', 'rank', 'details',
                                          'similar_item', 'date', 'price', 'imageURL', 'imageURLHighRes', 'also_buy', 'also_view'], axis=1)

    products_df = pd.read_csv(input_data + '/products.txt', sep='\t')
    valid_products = set(products_df.pid.unique())

    metaproduct_df = metaproduct_df[metaproduct_df.asin.isin(valid_products)]
    #Create i2kg.txt
    products_id = metaproduct_df['asin'].unique()
    product_id2new_id = {}
    entities = {}
    with open(input_data + "/i2kg_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["eid", "pid", "name", "entity"])
        for new_id, pid in enumerate(products_id):
            product_id2new_id[pid] = new_id
            entities[pid] = new_id
            writer.writerow([new_id, pid, pid, pid]) #No name so we put asin
    fo.close()

    columns = list(metaproduct_df.columns)
    columns.remove('asin')
    columns.remove('main_cat')
    relation_name2id = {}
    with open(input_data + "/r_map.txt", "w+") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["id", "kb_relation", "name"])
        new_rid = 0
        for relation in columns:
            writer.writerow([new_rid, relation, relation])
            relation_name2id[relation] = new_rid
            new_rid += 1
    fo.close()

    #Create kg_final.txt and e_map.txt
    entity_names = set()
    for col in columns:
        entity_name = col
        entity_names.add(entity_name)

    last_id = len(entities)
    triplets = []
    pid2category, pid2provider = {}, {}
    for entity_name in entity_names:
        for _, row in metaproduct_df.iterrows():
            pid = row['asin']
            curr_attributes = row[entity_name]
            if entity_name == 'category' and curr_attributes == "":
                pid2category[pid] = 'NA'
                continue
            if entity_name == 'brand' and curr_attributes == "":
                pid2provider[pid] = 'NA'
                continue

            if type(curr_attributes) == list:
                for entity in curr_attributes:
                    if len(entity.split(" ")) > 3: continue # Probably wrong data skip
                    if entity_name == 'category' and pid not in pid2category:
                        pid2category[pid] = entity
                    if entity_name == 'brand' and pid not in pid2provider:
                        pid2provider[pid] = entity
                    if entity in entities:
                        triplets.append([entities[pid], entities[entity], relation_name2id[entity_name]])
                    else:
                        entities[entity] = last_id
                        triplets.append([entities[pid], entities[entity], relation_name2id[entity_name]])
                        last_id += 1
            else:
                if entity_name == 'category' and pid not in pid2category:
                    pid2category[pid] = curr_attributes
                if entity_name == 'brand' and pid not in pid2provider:
                    pid2provider[pid] = curr_attributes
                if curr_attributes not in entities:
                    entities[curr_attributes] = last_id
                    triplets.append([entities[pid], entities[curr_attributes], relation_name2id[entity_name]])
                    last_id += 1
                else:
                    triplets.append([entities[pid], entities[curr_attributes], relation_name2id[entity_name]])

    #Create e_map.txt
    with open(input_data + "/e_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["eid", "name", "entity"])
        for entity_id, new_id in entities.items():
            writer.writerow([new_id, entity_id, entity_id])
    fo.close()

    #Create kg_final.txt
    with open(input_data + "/kg_final.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_head","relation","entity_tail"])
        for triple in triplets:
            e_h, e_t, r = triple
            triple = [e_h, r, e_t]
            writer.writerow(triple)
    fo.close()

    # Update products with providee
    products_df['name'] = products_df['pid']
    products_df['provider_id'] = products_df['pid'].map(pid2provider)
    products_df['genre'] = products_df['pid'].map(pid2category)
    products_df.to_csv(input_data + '/products.txt', sep='\t', index=False)

