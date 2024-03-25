
from typing import List, Dict
from typing import Optional, Tuple, Union
import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
class PERLM(GPT2LMHeadModel):
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.num_kg_types = len(PERLM.kg_categories)

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types,
                                                  embedding_dim=config.hidden_size)  # for entities, relations, and special tokens
        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()
        self.idx_mask_cache = dict()

    def __init_type_embeddings(self, batch_size, num_hops):
        # num_hops = self.config.num_hops
        n_tokens = num_hops  # num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size, n_tokens), dtype=torch.long)

        for i in range(n_tokens):
            if i == 0 or i == n_tokens - 1:
                type_ids[:, i] = PERLM.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:, i] = PERLM.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:, i] = PERLM.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def __get_type_embeds(self, n_rows, n_cols):
        row = self.type_ids_row[:, :(n_cols - 1)]
        row = torch.hstack([row, torch.ones((1, 1)) * PERLM.SPECIAL_ID])
        type_ids = torch.vstack([row for _ in range(n_rows)])
        return type_ids, self.type_embeddings(type_ids.to(self.type_embeddings.weight.device))

    def __get_even_idx_mask(self, n_rows, n_cols):
        mask_key = (n_rows, n_cols)
        cur_mask = self.even_idx_mask[:(n_cols)]
        return torch.vstack([cur_mask for _ in range(n_rows)])

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
        k = (batch_size, seq_len)
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids, type_embeds

        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]

        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

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

        sequence_output = transformer_outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + transformer_outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
