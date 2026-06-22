# Linxicon Solver

<p align="justify">
Linxicon Solver finds a "word chain" between two words, where each step in the
chain moves to a semantically similar word. Similarity is computed from
sentence embeddings rather than dictionary definitions, so the chain links
words that are *close in meaning*, not just spelled similarly.

You give it a start word and an end word, pick a search algorithm, and it
prints the path it found along with the similarity score between each
consecutive pair.
</p>

## How it works

1. **Vocabulary & embeddings** — On first run, the program downloads the
   NLTK English word list, filters it to words between 3 and 15 characters,
   and encodes every word with the `all-MiniLM-L6-v2` sentence-transformer
   model. The resulting embeddings are cached to disk (`vocab_embeddings.npy`,
   `vocab_words.npy`) so future runs load instantly instead of re-encoding.
2. **Similarity** — Two words' "closeness" is the cosine similarity between
   their embeddings.
3. **Linked words** — For any word, its neighbors are all other vocabulary
   words whose similarity exceeds a fixed threshold (`0.41`), sorted from
   most to least similar.
4. **Search** — Starting from word 1, the program searches the graph of
   linked words (using one of three algorithms below) until it reaches
   word 2, then reconstructs and prints the path.

## Algorithms

| Algorithm | File | Strategy |
|---|---|---|
| **UCS** (Uniform Cost Search) | `ucs.py` | Expands the cheapest path found so far (cost = accumulated `1 - similarity`). Guarantees the lowest-total-cost path. |
| **GBFS** (Greedy Best-First Search) | `gbfs.py` | Always expands the word that looks closest to the goal *right now* (no memory of accumulated cost). Fast, but not guaranteed optimal. |
| **A\*** | `astar.py` | Like UCS, but also uses a heuristic (estimated distance to the goal) to guide the search toward the target faster while still aiming for an optimal path. |

In all three, each node only considers its top connections (UCS and GBFS use
all words above the similarity threshold; A* additionally caps this to the
20 strongest links per word to keep the search fast).

## Requirements

- Python 3.8+
- Packages:
  ```
  pip install sentence-transformers nltk numpy scikit-learn
  ```

The first run will also trigger an `nltk.download('words')` call, which needs
an internet connection (the sentence-transformer model itself is loaded in
offline mode via `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`, so make sure the
model is already cached locally, e.g. via a prior online run or a manual
download).

## Usage

```bash
python linxicon_solver.py
```

You'll be prompted for:

```
Insert the 1st word: dog
Insert the 2nd word: cat
Choose your algorithm (UCS/GBFS/A*): ucs
```

Both words must exist in the program's vocabulary (lowercase English words,
3–15 letters) or you'll get an `INVALID WORD` message.

### Example output

```
======= LINXICON SOLVER =======
Insert the 1st word: dog
Insert the 2nd word: cat
Choose your algorithm (UCS/GBFS/A*): ucs

--- Starting at 'dog' (similarity to goal: 0.5512) ---
========== SOLUTION ==========
Path:
- dog
- puppy (87.34%)
- kitten (79.21%)
- cat (91.05%)
Average similarity: 85.87%
```

(Actual words and scores depend on the vocabulary and embeddings generated
on your machine.)

## Project structure

```
linxicon_solver.py   # Entry point: loads the model, takes input, runs the chosen algorithm
utils.py             # Vocabulary loading/caching, similarity & linked-word helpers
ucs.py               # Uniform Cost Search implementation
gbfs.py               # Greedy Best-First Search implementation
astar.py              # A* Search implementation
```
