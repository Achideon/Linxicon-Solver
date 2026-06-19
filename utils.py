import nltk
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def get_vocab(model):
    EMBEDDINGS_PATH = "vocab_embeddings.npy"
    WORDS_PATH = "vocab_words.npy"

    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading cached embeddings...")
        vocab_embeddings = np.load(EMBEDDINGS_PATH)
        word_list = np.load(WORDS_PATH, allow_pickle=True).tolist()
    else:
        nltk.download('words')
        word_list = [w.lower() for w in words.words() if 3 <= len(w) <= 15]
        print("Encoding vocabulary...")
        vocab_embeddings = model.encode(word_list, batch_size=512, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, vocab_embeddings)
        np.save(WORDS_PATH, word_list)
        print("Saved!")
    word_to_index = {word: i for i, word in enumerate(word_list)}
    return word_list, vocab_embeddings, word_to_index

def get_similarity(vocab_embeddings, word_to_index, word1, word2):
    e1 = vocab_embeddings[word_to_index[word1]]
    e2 = vocab_embeddings[word_to_index[word2]]
    score = cosine_similarity([e1], [e2])[0][0]
    return round(float(score), 4)

def get_linked_words(vocab_embeddings, word_to_index, word_list, word):
    threshold = 0.41
    keyword_embed = vocab_embeddings[word_to_index[word]]
    scores = cosine_similarity([keyword_embed], vocab_embeddings)[0]
    linked_indices = np.where(scores > threshold)[0]
    linked_indices = linked_indices[np.argsort(scores[linked_indices])[::-1]]
    results = []
    for i in linked_indices:
        if(word_list[i] != word):
            results.append((word_list[i], round(float(scores[i]), 4)))
    return results