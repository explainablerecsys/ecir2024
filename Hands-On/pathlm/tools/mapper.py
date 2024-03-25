import argparse
import pickle
from pathlm.knowledge_graphs.kg_macros import USER, PRODUCT, ENTITY
from transformers import PreTrainedTokenizerFast, set_seed,AutoConfig,AutoModelForCausalLM
from datasets import load_from_disk
from pathlm.utils import *
from pathlm.sampling import KGsampler








class EmbeddingMapper:
    def __init__(self, tokenizer, kg, embedding):
        self.tokenizer = tokenizer
        self.kg = kg
        self.embedding = embedding

        #for elem_type, glob_eid in kg.global_eid_to_cat_eid:        
        
        
    def get_embedding(self, token_id):
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        if not EmbeddingMapper.is_kg_element(token):
            return None
        kg_info = self.kg.dataset_info
        
        element_type = None
        for i, ch in enumerate(token):
            if not ch.isalpha():
                element_type = token[:i]
                break
        element_id = int(token[i:])
        if element_type == 'U':
            key = element_id # kg_info.groupwise_global_eid_to_cat_eid[USER][element_id]
            elem_type = 'user' #kg_info.groupwise_global_eid_to_subtype[USER][element_id]
        elif element_type == 'P':
            key = kg_info.groupwise_global_eid_to_cat_eid[PRODUCT][element_id]
            elem_type = kg_info.groupwise_global_eid_to_subtype[PRODUCT][element_id]
        elif element_type == 'R':
            key = 0
            elem_type = self.kg.rel_id2type[element_id]            
        else:
            key = kg_info.groupwise_global_eid_to_cat_eid[ENTITY][element_id]
            elem_type = kg_info.groupwise_global_eid_to_subtype[ENTITY][element_id]
        emb = self.embedding[elem_type][key]
        return emb
            
    
    def is_kg_element(token):
        return len(token) > 1 and token[0].isalpha() and token[1:].isnumeric()
        


    def init_with_embedding(self, wte_weights):
        wte_weights.requires_grad = False
        #elements = set()
        for token, token_id in self.tokenizer.get_vocab().items():
            if EmbeddingMapper.is_kg_element(token):
                #elements.add(token)
                # wte_weights is the tensor of embeddings obtained from the attribute "model.transformer.wte.weight"  
                #print('Before: ', wte_weights[token_id].data)
                wte_weights[token_id].copy_ (torch.FloatTensor(self.get_embedding(token_id)) )  
                #print('After: ', wte_weights[token_id].data)
                #print('Expected: ', self.get_embedding(token_id))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--context_length", type=int, default=100,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--emb_size", type=int, default=100,
                        help="Embedding size used to initialize the LM embeddings\n(must match the size of the embedding weights of the chosen external model)")

    args = parser.parse_args(args=[])

    return args

if __name__ == '__main__':

    args = get_arguments()
        

    set_seed(SEED)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    embedding_root_dir='./embedding-weights'

    dataset_filepath = f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf"

    data_dir_mapping= f'data/{dataset_name}/preprocessed/mapping/'
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    embed_filepath = os.path.join(embedding_root_dir, dataset_name, 'transe_embed.pkl')

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    tokenized_dataset = load_from_disk(dataset_filepath)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)



    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = get_data_dir(dataset_name)
    kg = KGsampler(dataset_name, data_dir=data_dir_mapping)


    
    #scorer=TransEScorer(dataset_name, embedding_root_dir)
    embeds = pickle.load(open(embed_filepath, 'rb'))

    mapper = EmbeddingMapper(tokenizer, kg, embeds)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    config = AutoConfig.from_pretrained(
        model_name,
        hidden_size=args.emb_size,
        num_attention_heads=args.emb_size//10,
        vocab_size=len(tokenizer),
        n_ctx=args.context_length,
        #n_positions=context_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = AutoModelForCausalLM.from_config(config)

    mapper.init_with_embedding(model.transformer.wte.weight)

