import pickle
import numpy as np
import os

class TransEScorer:

    def __init__(self, dataset_name, root_dir, device='cpu'):
        embed_filepath = os.path.join(root_dir, dataset_name, 'transe_embed.pkl')
        self.embeds = pickle.load(open(embed_filepath, 'rb'))
        #self.embed_size = self.embeds[USER].shape[1]
        for k in self.embeds:
            if isinstance(self.embeds[k],tuple):
                print(self.embeds[k][0].shape)
            else:
                print(k, self.embeds[k].shape)


    def score(self, cur_ent_t, cur_ent_id, next_ent_t, next_ent_id, rel, prev_score=1.):
        src_embed = self.embeds[cur_ent_t][cur_ent_id]
        rel_embed = self.embeds[rel][0]
        dest_embed = self.embeds[next_ent_t][next_ent_id]
        #print(rel, self.embeds[rel][0].flatten(), self.embeds[rel][1].flatten() )
        #print(src_embed, rel_embed)
        #print(dest_embed)
        score = np.matmul(src_embed + rel_embed[0], dest_embed  )
        #print(score, prev_score)
        #print()
        #print()
        return score  + prev_score         

        