print("Importing packages...")
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
print("SentenceTransformer imported")
from utils import get_vocab, get_similarity
from ucs import solve_ucs
from gbfs import solve_gbfs
from astar import solve_astar

def print_solution(vocab_embeddings, word_to_index, solution):
    print("\033[94m==========  SOLUTION ==========\033[0m")
    print("Path:")
    print("- " + solution[0])
    simsum = 0
    for index in range(1, len(solution)):
        sim = get_similarity(vocab_embeddings, word_to_index, solution[index-1], solution[index])
        print("- " + solution[index] + " \033[96m(" + str(sim * 100) + "%)\033[0m")
        simsum += sim
    print("Average similarity: \033[96m" + str(simsum / (len(solution)-1) * 100) + "%\033[0m")
    print()

def main():
    print("Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    word_list, vocab_embeddings, word_to_index = get_vocab(model)

    print()
    print("\033[94m======= LINXICON SOLVER =======\033[0m")
    word1 = input("\033[0mInsert the 1st word: \033[93m")
    word2 = input("\033[0mInsert the 2nd word: \033[93m")
    algo  = input("\033[0mChoose your algorithm (UCS/GBFS/A*): \033[93m")
    print("\033[0m")
    if word1 not in word_list:
        print("\033[91mINVALID 1ST WORD\033[0m")
        return
    if word2 not in word_list:
        print("\033[91mINVALID 2ND WORD\033[0m")
        return
    solution = []
    if algo.lower() == "ucs":
        solution = solve_ucs(word_list, vocab_embeddings, word_to_index, word1, word2)
    elif algo.lower() == "gbfs":
        solution = solve_gbfs(word_list, vocab_embeddings, word_to_index, word1, word2)
    elif algo.lower() == "astar":
        solution = solve_astar(word_list, vocab_embeddings, word_to_index, word1, word2)
    else:
        print("\033[91mINVALID ALGORITHM\033[0m")
        return
    # print(solution)
    print_solution(vocab_embeddings, word_to_index, solution)

if __name__ == "__main__":
    main()