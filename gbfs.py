import heapq
from utils import get_linked_words, get_similarity

def solve_gbfs(word_list, vocab_embeddings, word_to_index, word1, word2):
    visited = set()
    prio_que = []
    heapq.heappush(prio_que, 
                   (1 - get_similarity(vocab_embeddings, 
                                       word_to_index, 
                                       word1, word2), 
                    word1, [word1]))
    count = 0
    while prio_que:
        h, current, path = heapq.heappop(prio_que)
        if current == word2:
            return path
        if current not in visited:
            visited.add(current)
            linked = get_linked_words(vocab_embeddings, 
                                      word_to_index, 
                                      word_list, current)
            for word, score in linked:
                if word not in visited:
                    heapq.heappush(prio_que, 
                                   (1 - get_similarity(vocab_embeddings, 
                                                       word_to_index, 
                                                       word, word2), 
                                    word, path + [word]))
        count += 1
        if count % 100 == 0:
            print("--- Checked " + str(count) + "th word ---")                    
    return None
