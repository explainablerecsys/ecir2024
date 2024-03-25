import numpy as np
import torch
import torch.nn as nn

from pathlm.knowledge_graphs.kg_macros import USER, INTERACTION, PRODUCT
from pathlm.models.embeddings.kge_abstract import KnowledgeGraphEmbedding
from pathlm.models.embeddings.kge_utils import TRANSE, get_knowledge_derived_relations, save_embed, get_embedding_ckpt_rootdir


class TransE(KnowledgeGraphEmbedding):
    def __init__(self, args, dataloader):
        super().__init__(args, dataloader)
        self.name = TRANSE

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size

        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float64), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        #loss = self.compute_loss_lfm(batch_idxs)
        #assert loss == loss2
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        """
        regularizations = []

        user_idxs = batch_idxs[:, 0]
        product_idxs = batch_idxs[:, 1]
        knowledge_relations = get_knowledge_derived_relations(self.dataset_name)
        #print(knowledge_relations)

        # user + interaction -> product
        up_loss, up_embeds = self.neg_loss(USER, INTERACTION[self.dataset_name], PRODUCT, user_idxs, product_idxs)
        #print('A', up_loss)
        regularizations.extend(up_embeds)
        loss = up_loss

        i = 2
        for curr_rel in knowledge_relations:
            entity_name, curr_idxs = self.relation2entity[curr_rel], batch_idxs[:, i]
            # product + curr_rel -> curr_entity
            curr_loss, curr_embeds = self.neg_loss(PRODUCT, curr_rel, entity_name, product_idxs, curr_idxs)
            #print('B', curr_rel, curr_loss)
            if curr_loss is not None:
                regularizations.extend(curr_embeds)
                loss += curr_loss
            i+=1
        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
                #print('C', l2_loss)
            loss += self.l2_lambda * l2_loss

        return loss


    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        #print(entity_head_embedding)
        #print(entity_tail_embedding)
        relation_vec = getattr(self, relation)  # [1, embed_size]
        #print(relation, relation_vec)
        #print()
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return self.kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


    def kg_neg_loss(self, entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                    relation_vec, relation_bias_embed, num_samples, distrib):
        """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

        Args:
            entity_head_embed: Tensor of size [batch_size, embed_size].
            entity_tail_embed: Tensor of size [batch_size, embed_size].
            entity_head_idxs:
            entity_tail_idxs:
            relation_vec: Parameter of size [1, embed_size].
            relation_bias: Tensor of size [batch_size]
            num_samples: An integer.
            distrib: Tensor of size [vocab_size].

        Returns:
            A tensor of [1].
        """
        batch_size = entity_head_idxs.size(0)
        entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
        example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
        example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

        entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
        pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
        relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
        pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
        pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

        neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
        neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
        neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
        neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
        neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

        loss = (pos_loss + neg_loss).mean()
        return loss, [entity_head_vec, entity_tail_vec, neg_vec]

    def extract_embeddings(self, dataset, epochs):
        """Note that last entity embedding is of size [vocab_size+1, d]."""
        EMBEDDING_CKPT_DIR = get_embedding_ckpt_rootdir(self.dataset_name)
        model_file = f'{EMBEDDING_CKPT_DIR}/transe_model_sd_epoch_{epochs}.ckpt'
        print('Load embeddings', model_file)
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        embeds = {}
        for entity_name in dataset.entity_names:
            embeds[entity_name] = state_dict[f'{entity_name}.weight'].cpu().data.numpy()[:-1]

        embeds[INTERACTION[self.dataset_name]] = (
            state_dict[INTERACTION[self.dataset_name]].cpu().data.numpy()[0],
            state_dict[f'{INTERACTION[self.dataset_name]}_bias.weight'].cpu().data.numpy()
        )
        for relation_name in dataset.other_relation_names:
            embeds[relation_name] = (
                state_dict[f'{relation_name}'].cpu().data.numpy()[0],
                state_dict[f'{relation_name}_bias.weight'].cpu().data.numpy()
            )
        save_embed(self.dataset_name, self.name, embeds)