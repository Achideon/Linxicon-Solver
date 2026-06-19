import heapq
from utils import get_linked_words, get_similarity
from ucs import make_path

def solve_astar(word_list, vocab_embeddings, word_to_index, word1, word2):
    h_cost = {word1: 1-get_similarity(vocab_embeddings, 
                                      word_to_index, 
                                      word1, word2)}
    prio_que = [(h_cost[word1], word1)]
    visited = {word1: (0, None)}

    count = 0
    while prio_que:
        cur_f, cur_node = heapq.heappop(prio_que)
        cur_cost = cur_f - h_cost[cur_node]
        if cur_cost > visited[cur_node][0]:
            continue
        if cur_node == word2:
            return make_path(visited, word2)
        linked = get_linked_words(vocab_embeddings, 
                                  word_to_index, 
                                  word_list, cur_node)[:20]
        for word, score in linked:
            cost = 1 - score + cur_cost
            if word not in visited or cost < visited[word][0]:
                visited[word] = (cost, cur_node)
                if word not in h_cost:
                    h_cost[word] = 1 - get_similarity(vocab_embeddings, 
                                                      word_to_index, 
                                                      word, word2)
                heapq.heappush(prio_que, (cost + h_cost[word], word))
        count += 1
        if count % 100 == 0:
            print("--- Checked " + str(count) + "th word ---")
    return None