import argparse
import os
from typing import Tuple, Dict
from knowledge_graph_utils import *
from utils import ML1M, get_raw_kg_dir, get_data_dir

RELATION2PLAIN_NAME = {"http://dbpedia.org/ontology/cinematography": "cinematography_by_cinematographer",
                       "http://dbpedia.org/property/productionCompanies": "produced_by_prodcompany",
                       "http://dbpedia.org/property/composer": "composed_by_composer",
                       "http://purl.org/dc/terms/subject": "belong_to_category",
                       "http://dbpedia.org/ontology/starring": "starred_by_actor",
                       "http://dbpedia.org/ontology/country": "produced_in_country",
                       "http://dbpedia.org/ontology/wikiPageWikiLink": "related_to_wikipage",
                       "http://dbpedia.org/ontology/editing": "edited_by_editor",
                       "http://dbpedia.org/property/producers": "produced_by_producer",
                       "http://dbpedia.org/property/allWriting": "wrote_by_writter",
                       "http://dbpedia.org/ontology/director": "directed_by_director"
                       }

def remove_entites_with_different_relations(output_dir: str) -> None:
    """
    Removes KG entities with conflicting relations from the knowledge graph.
    """
    kg_triplets_df = pd.read_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t")
    n_triplets = kg_triplets_df.shape[0]
    kg_triplets_list = [list(a) for a in zip(kg_triplets_df.entity_head,
                                             kg_triplets_df.relation, kg_triplets_df.entity_tail)]
    kg_triplets_list.sort(key=lambda x: x[1])
    entity_tail_rel = {}
    valid_triplets = []
    for triplet in list(kg_triplets_list):
        entity_h, r, entity_t = triplet

        if entity_t not in entity_tail_rel:
            entity_tail_rel[entity_t] = r
        else:
            if entity_tail_rel[entity_t] != r:
                continue
        valid_triplets.append(triplet)
    kg_triplets_df = pd.DataFrame(valid_triplets, columns=["entity_head", "relation", "entity_tail"])

    #Propagate removal and reset eid
    i2kg_df = pd.read_csv(os.path.join(output_dir, "i2kg_map.txt"), sep="\t")
    valid_products = i2kg_df.eid.unique()
    valid_tails = kg_triplets_df.entity_tail.unique()
    entity_df = pd.read_csv(os.path.join(output_dir, "e_map.txt"), sep="\t")
    entity_df = entity_df[(entity_df.eid.isin(valid_tails)) | (entity_df.eid.isin(valid_products))]
    entity_df.rename({"eid": "old_eid"}, axis=1, inplace=True)
    entity_df.insert(0, "eid", list(range(entity_df.shape[0])))
    old_eid2new_eid = dict(zip(entity_df.old_eid, entity_df.eid))
    kg_triplets_df.entity_tail = kg_triplets_df.entity_tail.map(old_eid2new_eid)
    entity_df.drop("old_eid", axis=1, inplace=True)
    entity_df.to_csv(os.path.join(output_dir, "e_map.txt"), sep="\t", index=False)
    kg_triplets_df.to_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t", index=False)
    print(f"Removed {n_triplets - kg_triplets_df.shape[0]} triplets")

def standardize_rmap(kg_raw_folder: str, output_dir: str) -> Dict[int, int]:
    """
    Standardizes the relation map.
    """
    relations_df = pd.read_csv(os.path.join(kg_raw_folder, "r_map.txt"), sep="\t")

    # Wikipedia workaround (the entity in later stage of the process must be considered at last since many of its tails
    # overlap with other relations)
    row_index = 6
    wikipedia_rel_row = relations_df.iloc[[row_index]]
    relations_df = relations_df.drop(row_index)
    relations_df = pd.concat([relations_df, wikipedia_rel_row], ignore_index=True)

    relations_df.insert(0, "id", list(range(relations_df.shape[0])))
    old_rid2new_rid = dict(zip(relations_df.relation_id, relations_df.id))
    relations_df.rename({"relation_url": "kb_relation"}, axis=1, inplace=True)
    relations_df["name"] = relations_df.kb_relation.map(RELATION2PLAIN_NAME)
    relations_df = relations_df[["id", "kb_relation", "name"]]
    relations_df.to_csv(os.path.join(output_dir, "r_map.txt"), sep="\t", index=False)
    return old_rid2new_rid


def standardize_entities(
        output_dir: str,
        products_df: pd.DataFrame,
        i2kg_df: pd.DataFrame,
        e_map_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Standardizes entity mappings and updates entity references.

    Parameters:
    - output_dir: The directory to save standardized files.
    - products_df: DataFrame containing product information.
    - i2kg_df: DataFrame mapping dataset IDs to KG entity URLs.
    - e_map_df: DataFrame mapping entities to URLs and names.

    Returns:
    - Updated i2kg_df and e_map_df DataFrames, and a dictionary mapping old entity IDs to new entity IDs.
    """

    # Filter i2kg_df to include only items present in products_df
    i2kg_df_filtered = i2kg_df[i2kg_df.dataset_id.isin(products_df.pid)]

    # Identify unique entity IDs from the filtered i2kg DataFrame
    items_entity_ids = set(i2kg_df_filtered.entity_id.unique())

    # Extract entity names from URLs
    e_map_df["name"] = e_map_df.entity_url.apply(lambda x: x.split("/")[-1])

    # Separate entities into those related to items and others
    entity_items = e_map_df[e_map_df.entity_id.isin(items_entity_ids)]
    other_entities = e_map_df[~e_map_df.entity_id.isin(items_entity_ids)]

    # Combine and reindex entities
    all_entities_combined = pd.concat([entity_items, other_entities]).reset_index(drop=True)
    all_entities_combined.insert(0, "eid", range(len(all_entities_combined)))

    # Create a mapping from old entity IDs to new entity IDs
    old_eid_to_new_eid = dict(zip(all_entities_combined.entity_id, all_entities_combined.eid))

    # Update entity URLs to entity names and reformat DataFrame
    all_entities_combined = all_entities_combined[["eid", "name", "entity_url"]]
    all_entities_combined.rename(columns={"entity_url": "entity"}, inplace=True)

    # Save the updated entity map
    all_entities_combined.to_csv(os.path.join(output_dir, "e_map.txt"), sep="\t", index=False)

    # Update i2kg DataFrame to reflect standardized entity references
    i2kg_df_updated = i2kg_df_filtered[["dataset_id", "entity_url"]].rename(
        columns={"dataset_id": "pid", "entity_url": "entity"})
    entity_items_updated = pd.merge(all_entities_combined, i2kg_df_updated, on="entity")
    entity_items_updated.drop_duplicates(subset="name", inplace=True)
    entity_items_updated = entity_items_updated[["eid", "pid", "name", "entity"]]

    # Save the updated i2kg map
    entity_items_updated.to_csv(os.path.join(output_dir, "i2kg_map.txt"), sep="\t", index=False)

    return i2kg_df_updated, all_entities_combined, old_eid_to_new_eid

def preprocess_kg(args):
    """
    Perfoms the following steps:
    1. Standarize rmap to be from 0 to n_relations
    2. Standardize emap to be from 0 to n_entities
    3. Drop triplets that is involved in multiple triplets (leaving the first)
    """
    dataset_name = args.data
    kg_raw_folder = get_raw_kg_dir(dataset_name)
    preprocessed_folder = get_data_dir(dataset_name)

    #Stardadize r_map.txt
    old_rid2new_rid = standardize_rmap(kg_raw_folder, preprocessed_folder)

    #Standardize e_map.txt
    i2kg_df = pd.read_csv(os.path.join(kg_raw_folder, "i2kg_map.txt"), sep="\t")
    e_map_df = pd.read_csv(os.path.join(kg_raw_folder, "e_map.txt"), sep="\t")
    products_df = pd.read_csv(os.path.join(preprocessed_folder, "products.txt"), sep="\t")
    i2kg_df, e_map_df, old_eid2new_eid = standardize_entities(preprocessed_folder, products_df, i2kg_df, e_map_df)

    #Standardize kg_final.txt
    kg_triplets_df = pd.read_csv(os.path.join(kg_raw_folder, "kg_final.txt"), sep="\t")
    kg_triplets_df.entity_head = kg_triplets_df.entity_head.map(old_eid2new_eid)
    kg_triplets_df.entity_tail = kg_triplets_df.entity_tail.map(old_eid2new_eid)
    kg_triplets_df.relation = kg_triplets_df.relation.map(old_rid2new_rid)
    kg_triplets_df = kg_triplets_df.dropna()
    kg_triplets_df['entity_tail'] = pd.to_numeric(kg_triplets_df['entity_tail'])
    kg_triplets_df.to_csv(os.path.join(preprocessed_folder, "kg_final.txt"), sep="\t", index=False)

    remove_entites_with_different_relations(preprocessed_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ML1M, help='One of {ML1M, LFM1M}')
    args = parser.parse_args()

    preprocess_kg(args)

if __name__ == '__main__':
    main()
