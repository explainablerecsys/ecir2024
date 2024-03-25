from pathlm.knowledge_graphs.kg_macros import PRODUCT, USER, ENTITY, RELATION



class Trie:
    TERMINATION = ''
    class Item:
        def __init__(self, key, trie, counter):
            self.key = key
            self.trie = trie
            self.counter = counter

    def __init__(self, strings):
        
        self.trie = dict()
        self.strings = strings
        for string in strings:
            self.insert(string)
    def compute_unique_prefixes(self):
        prefix_map = dict()
        for word in self.strings:
            cur_trie = self.trie
            i = 0
            while i < len(word):
                cur_ch = word[i]
                i += 1
                if len(cur_trie[cur_ch].trie) == 1 and cur_trie[cur_ch].counter == 1:
                    break
                cur_trie = cur_trie[cur_ch].trie
            prefix_map[word] = word[:i]  
        return prefix_map


    def insert(self, word):
        cur_trie = self.trie
        for ch in word:
            if ch not in cur_trie:
                cur_trie[ch] = Trie.Item(ch, dict(), 0)
            cur_item = cur_trie[ch]
            cur_item.counter += 1
            cur_trie = cur_item.trie
        cur_trie[Trie.TERMINATION] = None 


class LiteralPath:
	
    main_user = ''#'Us'
    oth_user = ''#'U'
    ent = ''#'E'
    prod = ''#'P'

    user_type = 'U'
    prod_type = 'P'
    ent_type = 'E'
    rel_type = 'R'






    recom_prod = ''#'P'#'Ps'
    fw_rel = ''#'R' #'Rf'
    back_rel = ''#'R' #'Rb'
    interaction_rel_id = '-1'

    def get_ids(strings):
        TERMINATION = ''
        trie = dict()

        for string in strings:
            cur_trie = trie
            for ch in string:
                if ch not in cur_trie:
                    cur_trie[ch] = dict()
                cur_trie = cur_trie[ch]
            cur_trie[TERMINATION] = ''



class TypeMapper:
    mapping = { LiteralPath.user_type : USER,
            LiteralPath.prod_type : PRODUCT,
            LiteralPath.ent_type : ENTITY,
            LiteralPath.rel_type : RELATION,  }
    inv_mapping = { v:k for k,v in mapping.items() }
    
