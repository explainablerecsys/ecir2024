
from typing import List, Dict
from typing import Optional, Tuple, Union
import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2LMHeadModel, GPT2Model
class PLMRec(GPT2LMHeadModel):
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID] 

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        
        self.num_labels = config.vocab_size

        self.ent_mask = config.ent_mask
        self.rel_mask = config.rel_mask
        self.ent_mask_weight = torch.FloatTensor(self.ent_mask)
        self.rel_mask_weight = torch.FloatTensor(self.rel_mask)      
        self.ent_mask = torch.LongTensor(self.ent_mask)
        self.rel_mask = torch.LongTensor(self.rel_mask)      
        self.context_length = config.n_ctx

        
        self.num_kg_types = len(PLMRec.kg_categories)

        self.id_to_str = config.token_id_to_token

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types, embedding_dim=config.n_embd) # for entities, relations, and special tokens

        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()
        self.idx_mask_cache = dict()

        self.type_ids_row, self.type_embeds_row = self.__init_type_embeddings(1, self.context_length)

        # Create an additional linear layer for the second prediction head
        self.entity_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.relation_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def __init_type_embeddings(self,  batch_size, num_hops):
        #num_hops = self.config.num_hops
        n_tokens = num_hops#num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size,n_tokens) , dtype=torch.long)


        for i in range(n_tokens):
            if i == 0 or i == n_tokens-1:
                type_ids[:,i] = PLMRec.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:,i] = PLMRec.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:,i] = PLMRec.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def __get_type_embeds(self, n_rows, n_cols):
        row = self.type_ids_row[:,:(n_cols-1)]
        row = torch.hstack( [row, torch.ones((1,1))*  PLMRec.SPECIAL_ID  ] )
        type_ids = torch.vstack([row for _ in range(n_rows)]) 
        return type_ids, self.type_embeddings(type_ids.to(self.type_embeddings.weight.device))


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:


        batch_size, seq_len = input_ids.shape
        k = (batch_size, seq_len+1) 
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len+1)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids[:,:-1], type_embeds[:,:-1,:]
        
        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]

        
        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_entity_head.weight.device)


        # Get logits from the two heads, first based on entity tokens, then on relation tokens
        lm_entity_logits = self.entity_head(hidden_states)#[entity_token_ids])
        lm_relation_logits = self.relation_head(hidden_states)#[relation_token_ids])
        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]


        def compute_loss(logits, labels, class_mask=None):
            if class_mask is not None:
                class_mask = class_mask.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=class_mask)
            logits = logits.contiguous()
            labels = labels.contiguous()
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)) 
            return lm_loss  
        '''
        start,E,R,E,R,E,R,E,end

        start,E,R,E,R,E,end

        start,E,R,E,R,E
        E,R,E,R,E,end

        # ent
        start,R,R,end
        start,E,E,E,end  #(slice)
        start,R,R,end     # [:(-1)]
        E,E,E,end  #(slice) [1:-1]   labels       

        # rel
        start,E,E,E,end   #(slice) [1:-1]  preds 
        start,R,R,end      #(slice)[1:]   labels
        '''      
        loss = 0.
        # entity pred mask
        logits_mask = (type_ids != PLMRec.ENTITY_ID)[:, :(-1)].to(lm_entity_logits.device)#.unsqueeze(-1).expand(self.type_embeds.size())  
        label_mask = (type_ids != PLMRec.RELATION_ID)[:, 1:(-1)].to(lm_entity_logits.device)#.unsqueeze(-1)
        


        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]
        

        loss += compute_loss(lm_entity_logits[:,:(-1),:][logits_mask], input_ids[:,1:(-1)][label_mask],
                    self.ent_mask_weight)


        # relation pred mask
        logits_mask = (type_ids != PLMRec.RELATION_ID)[:,1:(-1)].to(lm_entity_logits.device)
        label_mask = (type_ids != PLMRec.ENTITY_ID)[:,1:].to(lm_entity_logits.device)
        

        loss += compute_loss(lm_relation_logits[:,1:(-1),:][logits_mask], input_ids[:,1:][label_mask],
                    self.rel_mask_weight)

        for i in range(sequence_len ):
            if i % 2 == 1:
                lm_entity_logits[:,i, :] = lm_relation_logits[:,i,:]  
        

        if not return_dict:
            output = (lm_entity_logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output                

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_entity_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
