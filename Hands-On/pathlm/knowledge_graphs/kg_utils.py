import json
from pathlm.knowledge_graphs.kg_macros import *

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

def get_KG_structures():
    with open(os.path.join(ROOT_DIR, 'pathlm/knowledge_graphs/config_files/kg_structure_config.json'), 'r') as f:
        KG_STRUCTURES = json.load(f)
    return KG_STRUCTURES
def get_KG_metapaths():
    with open(os.path.join(ROOT_DIR, 'pathlm/knowledge_graphs/config_files/kg_metapaths_config.json'), 'r') as f:
        KG_METAPATHS = json.load(f)
    return KG_METAPATHS

# 0 is reserved to the main relation, 1 to mention
PATH_PATTERN = get_KG_metapaths()
KG_RELATION = get_KG_structures()

MAIN_PRODUCT_INTERACTION = {
    ML1M: (PRODUCT, INTERACTION[ML1M]),
    LFM1M: (PRODUCT, INTERACTION[LFM1M]),
    CELL: (PRODUCT, PURCHASE),
    COCO: (PRODUCT, INTERACTION[COCO]),
}

def get_entities(dataset_name):
    return list(KG_RELATION[dataset_name].keys())

def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans

def get_dataset_relations(dataset_name, entity_head):
    return list(KG_RELATION[dataset_name][entity_head].keys())

def get_entity_tail(dataset_name, relation):
    entity_head, _ = MAIN_PRODUCT_INTERACTION[dataset_name]
    return KG_RELATION[dataset_name][entity_head][relation]
