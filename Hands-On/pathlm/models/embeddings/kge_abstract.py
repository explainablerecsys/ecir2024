from __future__ import absolute_import, division, print_function

import torch.nn as nn
from easydict import EasyDict as edict

from pathlm.knowledge_graphs.kg_macros import INTERACTION, PRODUCT


class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        self.dataset_name = args.dataset
        self.relation_names = dataloader.dataset.other_relation_names
        self.entity_names = dataloader.dataset.entity_names
        self.relation2entity = dataloader.dataset.relation2entity

        # Initialize entity embeddings.
        self.initialize_entity_embeddings(dataloader.dataset)

        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.initialize_relations_embeddings(dataloader.dataset)

        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def initialize_entity_embeddings(self, dataset):
        self.entities = edict()
        for entity_name in self.entity_names:
            value = edict(vocab_size=getattr(dataset, entity_name).vocab_size)
            self.entities[entity_name] = value

    def initialize_relations_embeddings(self, dataset):
        self.relations = edict()
        main_rel = INTERACTION[dataset.dataset_name]
        self.relations[main_rel] = edict(
            et=PRODUCT,
            et_distrib=self._make_distrib(getattr(dataset, "review").product_uniform_distrib)
        )
        for relation_name in dataset.other_relation_names:
            value = edict(
                et=dataset.relation2entity[relation_name],
                et_distrib=self._make_distrib(getattr(dataset, relation_name).et_distrib)
            )
            self.relations[relation_name] = value

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        raise NotImplementedError

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        raise NotImplementedError

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        raise NotImplementedError

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        raise NotImplementedError

    def forward(self, batch_idxs):
        raise NotImplementedError

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        """
        raise NotImplementedError

    def extract_embeddings(self, args, dataset):
        raise NotImplementedError