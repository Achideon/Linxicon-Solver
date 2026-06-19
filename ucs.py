import heapq
from utils import get_linked_words, get_similarity

def make_path(visited, word2):
    path = []
    current = word2
    while current is not None:
        path.append(current)
        current = visited[current][1]
    return path[::-1]

def solve_ucs(word_list, vocab_embeddings, word_to_index, word1, word2):
    prio_que = [(0, word1)]
    visited = {word1: (0, None)}
    best_similarity = get_similarity(vocab_embeddings, word_to_index, word1, word2)
    print(f"--- Starting at '{word1}' (similarity to goal: {best_similarity}) ---")

    count = 0
    while prio_que:
        cur_cost, cur_node = heapq.heappop(prio_que)
        word_similarity = get_similarity(vocab_embeddings, word_to_index, cur_node, word2)
        if word_similarity > best_similarity:
            best_similarity = word_similarity
            print(f"--- Current closest word: \033[92m'{word}'\033[0m (similarity to goal: \033[96m{best_similarity}\033[0m) ---")
        if cur_cost > visited[cur_node][0]:
            continue
        if cur_node == word2:
            return make_path(visited, word2)

        linked = get_linked_words(vocab_embeddings, word_to_index,
                                  word_list, cur_node)
        for word, score in linked:
            cost = 1 - score + cur_cost
            if word not in visited or cost < visited[word][0]:
                visited[word] = (cost, cur_node)
                heapq.heappush(prio_que, (cost, word))

        count += 1
        if count % 100 == 0:
            print("--- Checked " + str(count) + "th word ---")

    return None