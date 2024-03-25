
from collections import defaultdict
from typing import List

def visit(node):
    t = node
    while t != None:
        print(t.item,  end= ' ')
        t = t.next
    print()
    return
class Node:
    def __init__(self, item, prev=None,next=None):
        self.item = item
        self.prev = prev
        self.next = next


class UnboundedCache:
    def __init__(self):
        self.cache = dict()

    def put(self, key, value):
        if key not in self.cache:
            self.cache[key] = value
        
    def get(self, key):
        if key not in self.cache:
            return None 
        return self.cache[key]
    
class LFUCache:

    def __init__(self, capacity: int):
        
        self.k_freq = defaultdict(int)

        self.k_v = dict()
        
        self.k_node = dict()

        self.fk_ll_head = dict()
        self.fk_ll_tail = dict()

        self.f_ll_head = None

        self.freq_node = dict()
        self.f_count = defaultdict(int)
        self.capacity = capacity
        self.n = 0

    def get(self, key):

        if key not in self.k_v or self.capacity == 0:
            return None
        self.increase_freq(key)
        return self.k_v[key]


    def pop(self):
        lowest_frequency = self.f_ll_head.item

        h_node = self.fk_ll_head[lowest_frequency]

        self.f_count[lowest_frequency] -= 1
        if self.f_count[lowest_frequency] == 0:
            del self.f_count[lowest_frequency]
            del self.freq_node[lowest_frequency]

            del self.fk_ll_head[lowest_frequency]
            del self.fk_ll_tail[lowest_frequency]
            self.f_ll_head = self.f_ll_head.next
            if self.f_ll_head is not None:
                self.f_ll_head.prev = None

        else:
           
            h_next = h_node.next
            self.fk_ll_head[lowest_frequency] = h_next
            h_next.prev = None

        del self.k_freq[h_node.item]
        del self.k_v[h_node.item]
        del self.k_node[h_node.item]
        self.n -= 1

        


    def increase_freq(self, key):
        node = self.k_node[key]
        
        cur_freq = self.k_freq[key]
        del self.k_freq[key]
        self.k_freq[key] = cur_freq+1
        # remove from freq_bucket linkedlist
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        
        
        if node.prev is None:
            self.fk_ll_head[cur_freq] = node.next
        if node.next is None:
            self.fk_ll_tail[cur_freq] = node.prev
        if self.fk_ll_head[cur_freq] is None and self.fk_ll_tail[cur_freq] is None:
            del self.fk_ll_head[cur_freq]
            del self.fk_ll_tail[cur_freq]
   

        # remove from freq linkedlist
        self.f_count[cur_freq] -= 1
        cur_freq_node = self.freq_node[cur_freq]
        
        if self.f_count[cur_freq] == 0:

            del self.freq_node[cur_freq]
            del self.f_count[cur_freq]
            if cur_freq_node.next is not None:
                cur_freq_node.next.prev = cur_freq_node.prev
                if cur_freq_node.prev is not None:
                    cur_freq_node.prev.next = cur_freq_node.next                

            if cur_freq_node.prev is not None:
                cur_freq_node.prev.next = cur_freq_node.next
                if cur_freq_node.next is not None:
                    cur_freq_node.next.prev = cur_freq_node.prev
            else:
                self.f_ll_head = cur_freq_node.next
            
        
        # add to increased freq linkedlist
        if (cur_freq+1) not in self.f_count:
            preceding = cur_freq_node
            following = cur_freq_node.next
            if cur_freq not in self.f_count:
                preceding = cur_freq_node.prev            
            new_node = Node(cur_freq+1, preceding, following)
            self.freq_node[cur_freq+1] = new_node

            if preceding is not None:
                preceding.next = new_node
            else:
                self.f_ll_head = new_node
            if following is not None:
                following.prev = new_node


    
        self.f_count[cur_freq+1] += 1
        # add to increased freq bucket linkedlist
        node.next = None
        node.prev = None

        if (cur_freq+1) not in self.fk_ll_tail:
            self.fk_ll_tail[cur_freq+1] = node
            self.fk_ll_head[cur_freq+1] = node
        else:
            node.prev = self.fk_ll_tail[cur_freq+1]
            self.fk_ll_tail[cur_freq+1].next = node
            self.fk_ll_tail[cur_freq+1] = node

        
        

    def put(self, key, value) -> None:
        if self.capacity == 0:
            return

        if key not in self.k_v:
            self.k_v[key] = value
            node = None
            if self.n >= self.capacity:
                # do remove lfu for new key
                self.pop()
            node = Node(key)
            self.k_node[key] = node
            base_freq = 0
            self.k_freq[key] = base_freq
       
            if base_freq not in self.f_count:
                cur = Node(base_freq)
                self.freq_node[base_freq] = cur
                if self.f_ll_head is not None:
                    self.f_ll_head.prev = cur
                cur.next = self.f_ll_head
                self.f_ll_head = cur 

            self.f_count[base_freq] += 1
                
            self.n += 1
        self.k_v[key] = value
                
        
        self.increase_freq(key)




class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            





class LRUCache():
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = {}
        self.size = 0
        self.capacity = capacity
        self.head, self.tail = DLinkedNode(), DLinkedNode()


        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node):
        """
        Always add the new node right after head.
        """
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """
        Remove an existing node from the linked list.
        """
        prev = node.prev
        new = node.next

        prev.next = new
        new.prev = prev

    def _move_to_head(self, node):
        """
        Move certain node in between to the head.
        """
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        """
        Pop the current tail.
        """
        res = self.tail.prev
        self._remove_node(res)
        return res


        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        node = self.cache.get(key, None)
        if not node:
            return None

        # move the accessed node to the head;
        self._move_to_head(node)

        return node.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        node = self.cache.get(key)

        if not node: 
            newNode = DLinkedNode()
            newNode.key = key
            newNode.value = value

            self.cache[key] = newNode
            self._add_node(newNode)

            self.size += 1

            if self.size > self.capacity:
                # pop the tail
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            # update the value.
            node.value = value
            self._move_to_head(node)
